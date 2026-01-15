import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
from hash_encoding import HashEmbedder
from instant_ngp_model import NeRF_Instant
device='cuda' if torch.cuda.is_available() else 'cpu'
model=NeRF_Instant().to(device)
model.load_state_dict(torch.load('instant_ngp_trained_model.pth',map_location=device))
def get_query_points(min_bound,max_bound,res):

    x=torch.linspace(min_bound,max_bound,res)
    y=x
    z=x
    X,Y,Z=torch.meshgrid(x,y,z,indexing='ij')
    co_ordinates=torch.stack([X.flatten(),Y.flatten(),Z.flatten()],axis=-1)
    return co_ordinates
def mining_model(model,chunk_size,query_points):
    """
    Docstring for mining_model
    (x,y,z)-->model-->density
    but for high res no of (x,y,z) is high so we chunk it, then give it to model,model gives the density for that chunk,then 
    concatenate the chunk densities
    """
    density_list=[]
    for k in range(0,query_points.shape[0],chunk_size):
        co_ordinates_chunked=query_points[k:k+chunk_size].to(device)
        h=model.encoder(co_ordinates_chunked)
        geo_feat = model.density_net(h)
        sigma = F.softplus(geo_feat[..., 0]) # Density must be positive
        sigma=sigma.cpu()
        density_list.append(sigma)
    return torch.cat(density_list,dim=0)
res=128
query_points=get_query_points(min_bound=-2,max_bound=2,res=res)
map=mining_model(model,8192,query_points)
threshold=10
bool_array=(map>threshold).float()
bool_array=bool_array.reshape(res,res,res).detach().numpy()
np.save("map.npy",bool_array)
print("save")
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

# ax.voxels expects a boolean array
ax.voxels(bool_array > 0.1, edgecolor='k')

plt.title("Drone Navigation Map (Voxel Grid)")
plt.show()
