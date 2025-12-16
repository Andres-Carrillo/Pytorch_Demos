from training_log import TrainingLog
from torch.utils.data import DataLoader
from torch import nn
import torch
import os
import logging 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def classes_to_labels(classes: list[int],label_map:dict) -> list[str]:
    return [label_map[idx] for idx in classes]

class Trainer:
    def __init__(self,labels:dict[int,str],max_epochs:int=16,device:str='cpu')->None:
        self.logger = TrainingLog()
        self.labels = labels
        self.device = device
        self.epoch_number = 0
        self.max_epochs = max_epochs

    def __call__(self,dataloaders:tuple[DataLoader,DataLoader],model:nn.Module,
        loss_fn:nn.Module,optimizer:torch.optim.Optimizer,logging:bool=True)->None:

        for e in range(self.max_epochs):
            self._training_loop(dataloaders[0],model,loss_fn,optimizer,logging)
            self._validation_loop(dataloaders[1],model,loss_fn,logging)

    
    def save_logs(self,save_dir:str='training_logs'):
        self.logger.generate_graph('training','loss','batch','training_logs')
        self.logger.generate_graph('training','loss','epoch','training_logs')
        self.logger.generate_graph('test','avg_loss','epoch','training_logs')
        self.logger.generate_graph('test','accuracy','epoch','training_logs')
        self.logger.generate_graph('test','precision','epoch','training_logs')
        self.logger.generate_graph('test','f_score','epoch','training_logs')
        self.logger.generate_graph('test','recall','epoch','training_logs')
        self.logger.make_confusion_mat_graph(
            labels=list(self.labels.values()), 
            figsize=(10, 8),  # Optional - uses default if not specified
            save_dir=os.path.join(save_dir,'confusion_matrices')
        )


    def _training_loop(self,dataloader:DataLoader,model:nn.Module,
        loss_fn:nn.Module,optimizer:torch.optim.Optimizer,is_logging:bool=True) -> None:
        num_batches = len(dataloader)
        model.train()
        size = len(dataloader.dataset)
        epoch_loss = 0

        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(self.device), y.to(self.device)

            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            epoch_loss += loss.to('cpu').item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)


                if is_logging:
                    logging.info(f"loss: {loss}  [{current:>5d}/{size:>5d}]")
                    id = f"batch_[{self.epoch_number}]_[{current}]_loss"
                    self.logger.add_metric('training',id,loss)

        if is_logging:
            avg_epoch_loss = epoch_loss/num_batches
            self.logger.add_metric('training',f"epoch_[{self.epoch_number}]_loss",avg_epoch_loss)

    
    def _validation_loop(self,loader:DataLoader,model:nn.Module,loss_fn:nn.Module,is_logging:bool = True):
        self.epoch_number += 1
        size = len(loader.dataset)
        num_batches = len(loader)
        model.eval()
        test_loss,correct = 0,0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for X,y in loader:
                X,y = X.to(self.device),y.to(self.device)
                pred = model(X)
                test_loss += loss_fn(pred,y)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

                all_labels.append(y.cpu())
                all_predictions.append(pred.cpu())
            
            test_loss /= num_batches
            correct /= size



            if is_logging:
                logging.info(f"\ncurrent epoch:{self.epoch_number}")
                logging.info(f"\nAccuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
                id = f"epoch_[{self.epoch_number}]"
                epoch_preds = torch.cat(all_predictions,dim=0)
                epoch_labels = torch.cat(all_labels,dim=0).numpy()
                pred_probs = torch.softmax(epoch_preds,dim=1).numpy()
                epoch_pred_classes = torch.argmax(epoch_preds, dim=1).numpy()

                ground_truth_labels = classes_to_labels(epoch_labels,self.labels)
                predicted_labels = classes_to_labels(epoch_pred_classes,self.labels)

                self.logger.add_confusion_matrix('test',id,ground_truth_labels,predicted_labels,list(self.labels.values()))
                self.logger.calculate_metrics(self.epoch_number,epoch_labels,pred_probs)
                self.logger.add_metric('test',id + '_accuracy',correct)
                self.logger.add_metric('test',id + '_avg_loss',test_loss.item())

    
            