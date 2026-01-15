import torch
import torch.nn as nn
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class HashEmbedder(nn.Module):
    def __init__(self,room_min,room_max,no_of_levels=16,no_of_features=2,res_base=16,res_fine=512,log_2_hash_map_size=19):
        super().__init__()
        self.register_buffer('room_min',torch.tensor(room_min))
        self.register_buffer('room_max',torch.tensor(room_max))
        self.register_buffer('no_of_levels',torch.tensor(no_of_levels))
        self.register_buffer('no_of_features',torch.tensor(no_of_features))
        self.register_buffer('res_base',torch.tensor(res_base))
        self.register_buffer('res_fine',torch.tensor(res_fine))
        self.hashmap_size=2**log_2_hash_map_size
        self.register_buffer('primes',torch.tensor([1,2654435761,805459861],dtype=torch.int64))
        growth_factor=torch.exp((torch.log(torch.tensor(res_fine))-torch.log(torch.tensor(res_base)))/(torch.tensor(no_of_levels-1)))
        self.register_buffer('b',growth_factor)
        self.embeddings=nn.Parameter(nn.init.uniform(torch.empty(self.hashmap_size,no_of_features),a=-0.0001,b=0.0001))


    def forward(self,x):
        x_norm=(x-self.room_min)/(self.room_max-self.room_min)
        output_features=[]
        for i in range(self.no_of_levels):
            res=torch.floor(self.res_base*(self.b**i))
            x_scaled=x_norm*res
            x_not=torch.floor(x_scaled).int()
            weights=x_scaled-x_not.float()
            wx,wy,wz=weights[:,0],weights[:,1],weights[:,2]

            def get_corner_parameters(dx,dy,dz):
                c=x_not+torch.tensor([dx,dy,dz],device=x.device)
                h=(c[:,0]*self.primes[0])^(c[:,1]*self.primes[1])^(c[:,2]*self.primes[2])
                h=h%self.hashmap_size
                return self.embeddings[h]
            c000=get_corner_parameters(0,0,0)
            c001=get_corner_parameters(0,0,1)
            c010=get_corner_parameters(0,1,0)
            c011=get_corner_parameters(0,1,1)
            c100=get_corner_parameters(1,0,0)
            c101=get_corner_parameters(1,0,1)
            c110=get_corner_parameters(1,1,0)
            c111=get_corner_parameters(1,1,1)
            c00=c000*(1-wx).unsqueeze(-1)+c100*wx.unsqueeze(-1)
            c01=c001*(1-wx).unsqueeze(-1)+c101*wx.unsqueeze(-1)
            c10=c010*(1-wx).unsqueeze(-1)+c110*wx.unsqueeze(-1)
            c11=c011*(1-wx).unsqueeze(-1)+c111*wx.unsqueeze(-1)
            c0=c00*(1-wy).unsqueeze(-1)+c10*wy.unsqueeze(-1)
            c1=c01*(1-wy).unsqueeze(-1)+c11*wy.unsqueeze(-1)
            c=c0*(1-wz).unsqueeze(-1)+c1*wz.unsqueeze(-1)
            output_features.append(c)
        return torch.cat(output_features,dim=-1)
    
encoder=HashEmbedder(-2,2)
print(encoder(torch.tensor([[1,2,3]])).shape)

