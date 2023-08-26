import torch.nn as nn
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel,self).__init__()
        self.net1 = nn.Linear(10,10)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(10,5)
    
    def forward(self,x):
        net1_res = self.net1(x)
        relu_res = self.relu(net1_res)
        net2_res = self.net2(relu_res)
        return relu_res