import matplotlib.pyplot as plt
from torch import Tensor
from datetime import datetime
import os
import torch
from torch import nn
import logging 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def graph_data(x_axis:list,y_axis:list,fig_size:tuple = (10,6),
               title:str = "title",x_axis_title:str='x-axis',y_axis_title:str = 'y-axis',save_path:str=None) ->None:
    plt.figure(figsize=fig_size)
    plt.plot(x_axis,y_axis)

    plt.title(title)
    plt.xlabel(x_axis_title)
    plt.ylabel(y_axis_title)
    plt.grid(True, alpha=0.3)

    if save_path:
        dir = os.path.dirname(save_path)
        if dir:
            os.makedirs(dir,exist_ok=True)

            plt.savefig(save_path,dpi=300,bbox_inches='tight')
            plt.close()
    else:
        plt.show()

def save_dataset_sample(training_data:Tensor,labels:dict,save_path:str,dataset_name:str,row_count:int = 3,col_count:int = 3) ->None:
    figure = plt.figure()
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = dataset_name + time_stamp+ ".png"
    file_path = os.path.join(save_path,file_name)
    
    os.makedirs(save_path,exist_ok=True)
    
    for i in range(1,row_count*col_count + 1):
        # select random index
        sample_idx =torch.randint(len(training_data),size=(1,)).item()

        # grab both item and label
        img,label = training_data[sample_idx]

        figure.add_subplot(row_count,col_count,i)
        plt.title(labels[label])
        plt.axis("off")
        plt.imshow(img.squeeze(),cmap="gray")

    plt.savefig(file_path)
    plt.close(figure)

def save_model(model:nn.Module,directory_path:str,model_name:str)->None:
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S") +'.pth'
    full_path = os.path.join(directory_path,model_name +"_" + time_stamp)
    torch.save(model.state_dict(),full_path)

def now()->str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")