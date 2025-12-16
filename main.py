import torch
import random
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from model_zoo import FashionMNISTModel
from torch import nn
import logging 
from utils import save_dataset_sample
from training_log import TrainingLog
from trainer import Trainer
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

LABELS_MAP = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

def classes_to_labels(classes: list[int],label_map:dict) -> list[str]:
    return [label_map[idx] for idx in classes]

if __name__ == "__main__":
    training_data = datasets.FashionMNIST(root="data",train=True,download=True,transform=ToTensor())
    test_data = datasets.FashionMNIST(root="data",train=False,download=True,transform=ToTensor())

    batch_size = 64
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    save_dataset_sample(training_data,LABELS_MAP,"./dataset_samples",'FashionMINST')

    model = FashionMNISTModel((28,28),10).to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
    epochs = 64
    model_log = TrainingLog() 
    trainer = Trainer(LABELS_MAP,epochs,torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu")

    trainer((train_dataloader,test_dataloader),model,loss,optimizer)
    trainer.save_logs('training_logs')

    print("Done Training")