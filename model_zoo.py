import torch.nn as nn
from torch import Tensor

class FashionMNISTModel(nn.Module):
    def __init__(self,input_size:tuple,output_size:int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size[0] * input_size[1],512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,output_size)
        )

    def forward(self,X:Tensor) -> Tensor:
        x = self.flatten(X)
        x = x.to(next(self.parameters()).device)
        logits = self.linear_relu_stack(x)

        return logits
   

# class ConvNet(nn.Module):
#     def __init__(self)->None:
#         super().__init__()
#         self.linear_convnet = nn.Sequential(
#             nn.Conv2d(3,6,5),
#             nn.MaxPool2d(2,2),

#         )