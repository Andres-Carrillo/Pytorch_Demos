from dataclasses import dataclass, field
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,confusion_matrix
from enum import Enum
import os
import numpy as np
import seaborn as sns
from torch import Tensor
from utils import graph_data
import matplotlib.pyplot as plt

class LOG_ITEM(Enum):
    METRICS = 0
    EMBEDDINGS = 1
    CONFUSION_MATRIX = 2

@dataclass
class TrainingLog:
    metrics:dict[str,dict[str,list[float]]] = field(default_factory=dict)
    embeddings:dict[str,dict[str,list[Tensor]]] = field(default_factory=dict)
    confusion_matrices:dict[str,np.ndarray] = field(default_factory=dict)

    def add_metric(self,phase:str,metric:str,value:float) -> None:
        self._get_cache(LOG_ITEM.METRICS,phase,metric).append(value)

    def add_embeddings(self,phase:str,layer:str,embedding:Tensor) -> None:
        self._get_cache(LOG_ITEM.EMBEDDINGS,phase,layer,embedding).append(embedding)

    def add_confusion_matrix(self,phase:str,epoch:int,ground_truth:np.ndarray,predictions:np.ndarray,labels:list[str]) -> None:
        conf_mat = confusion_matrix(ground_truth,predictions,labels=labels)
        self.confusion_matrices[epoch] = conf_mat

    def _get_cache(self,cache_type:LOG_ITEM,phase,key) -> list:
        """Returns the correct class variable for adding to the internal logs"""
        match cache_type:
            case LOG_ITEM.METRICS:
                return self.metrics.setdefault(phase, {}).setdefault(key, [])
            case LOG_ITEM.EMBEDDINGS:
                return self.embeddings.setdefault(phase, {}).setdefault(key, [])
            case LOG_ITEM.CONFUSION_MATRIX:
                return self.confusion_matrices.setdefault(key, np.ndarray) # confusion matrices are only stored during testing
            case _:
                raise ValueError(f"Unsupported cache type: {cache_type}")
            
    def generate_graph(self, phase: str, metric_name:str='loss', level:str ="batch",save_dir: str = None) -> None:
        """Generates a graph using matplot lib and saves it if save_dir is not none"""
        if phase not in self.metrics:
            raise ValueError(f"phase: {phase} not found in log")
        
        phase_metrics = self.metrics[phase]
        loss_metrics = {key: values for key, values in phase_metrics.items() if metric_name.lower() in key.lower()}

        # Check for empty metrics FIRST
        if not loss_metrics:
            raise ValueError(f"No metrics containing '{metric_name}' found in phase '{phase}'")

        batch_losses = {key: values for key, values in loss_metrics.items() if level in key.lower()}
        data = [values for key, values in batch_losses.items() if values]        
        domain = list(range(1,len(data) + 1))

        if save_dir:
            save_path = os.path.join(save_dir,phase + '_' + metric_name + '_' + level +'_graph.png')
            graph_data(domain,data,title= phase + ' ' + metric_name,x_axis_title=level,y_axis_title=metric_name,save_path=save_path)
        else:
            graph_data(domain,data,title= phase + ' ' + metric_name,x_axis_title=level,y_axis_title=metric_name,save_path=save_dir)

    def calculate_metrics(self,epoch_number:int,labels:list,pred_prob:Tensor,multi_class:str='ovr',avg:str='weighted') ->dict:
        id = f'epoch[{epoch_number}]'
        epoch_pred_classes = np.argmax(pred_prob,axis=1)

        roc_auc = roc_auc_score(labels,pred_prob,multi_class=multi_class,average=avg)
        precision = precision_score(labels,epoch_pred_classes,average=avg,zero_division=0)
        recall = recall_score(labels,epoch_pred_classes,average=avg,zero_division=0)
        f_score = f1_score(labels,epoch_pred_classes,average=avg,zero_division=0)

        self.add_metric('test',id + '_roc_auc',roc_auc)
        self.add_metric('test',id + '_precision',precision)
        self.add_metric('test',id + '_recall',recall)
        self.add_metric('test',id +'_f_score',f_score)

    def make_confusion_mat_graph(self,labels:str,figsize:tuple = (10,8),save_dir:str=None) ->None:
        if not self.confusion_matrices:
            raise ValueError("no confusion matrices stored")
        
        if save_dir:
            os.makedirs(save_dir,exist_ok=True)

        for id,matrix in self.confusion_matrices.items():
            plt.figure(figsize=figsize)

            sns.heatmap(matrix,annot=True,fmt='d',cmap='Blues',xticklabels=labels,yticklabels=labels,cbar={'label':'Count'})
            
            plt.title(f'Confusion Matrix - {id}')
            plt.xlabel('Predicted Labels')
            plt.ylabel('True Labels')
            plt.tight_layout()
            
            # Save or show
            if save_dir:
                filename = f"confusion_matrix_{id}.png"
                save_path = os.path.join(save_dir, filename)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()