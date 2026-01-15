import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from instant_ngp_model import NeRF_Instant
from instant_ngp_component import get_rays,render_rays
from instant_ngp_data import load_blender_data

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#config
BASE_DIR='/content/lego/lego'
LR=1e-2
BATCH_SIZE=4096
if os.path.exists(os.path.join(BASE_DIR, 'transforms_train.json')):
    images_np, poses_np, H, W, focal = load_blender_data(BASE_DIR)
    
    images = torch.tensor(images_np).to(device) 
    poses = torch.tensor(poses_np).to(device)
    if images.shape[-1] == 4:
        images = images[..., :3] * images[..., 3:] + (1. - images[..., 3:])

else:
    print('data not found')
model = NeRF_Instant(bound=2.0).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-15)

for i in tqdm(range(5000)):
    img_i = np.random.randint(images.shape[0])
    target_img = images[img_i]
    pose = poses[img_i]
    coords = torch.stack(torch.meshgrid(
        torch.linspace(0, H-1, H), torch.linspace(0, W-1, W), indexing='xy'
    ), -1).reshape(-1, 2).long()
    
    select_inds = np.random.choice(coords.shape[0], size=[BATCH_SIZE], replace=False)
    select_coords = coords[select_inds]
    
    rays_o, rays_d = get_rays(H, W, focal, pose)
    rays_o = rays_o[select_coords[:, 1], select_coords[:, 0]]
    rays_d = rays_d[select_coords[:, 1], select_coords[:, 0]]
    target_rgb = target_img[select_coords[:, 1], select_coords[:, 0]]
    rgb_map = render_rays(model, rays_o, rays_d)
    loss = F.mse_loss(rgb_map, target_rgb)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 500 == 0:
        print(f"Step {i} | Loss: {loss.item():.4f}")
torch.save(model.state_dict(), 'instant_ngp_trained_model.pth')