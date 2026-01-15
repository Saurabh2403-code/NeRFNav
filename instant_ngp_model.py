from hash_encoding import HashEmbedder
import torch
import torch.nn as nn
import torch.nn.functional as F
class NeRF_Instant(nn.Module):
    def __init__(self,bound=2.0):
        super().__init__()
        self.room_min=[-bound,-bound,-bound]
        self.room_max=[bound,bound,bound]
        self.encoder=HashEmbedder(self.room_min,self.room_max)
        input_dim=16*2
        self.density_net=nn.Sequential(
            nn.Linear(input_dim,64),
            nn.ReLU(),
            nn.Linear(64,16)
        )
        self.color_net=nn.Sequential(
            nn.Linear(15+27,64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.ReLU(),
            nn.Linear(64,3)
        )
    def forward(self,x,d):
        x=self.encoder(x)
        out1=self.density_net(x)
        sigma=F.softplus(out1[:,0]).unsqueeze(-1)
        out2=out1[:,1:]
        dir_co_ordinates=torch.cat([d]+[torch.sin((2**i)*d) for i in range(4)]+[torch.cos((2**i)*d) for i in range(4)],dim=-1)
        input_2=torch.cat([out2,dir_co_ordinates],dim=-1)
        color=torch.sigmoid(self.color_net(input_2))
        return torch.cat([color,sigma],dim=-1)
