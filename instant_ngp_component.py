import torch
device=torch.device('cuda' if torch.cuda.is_available else 'cpu')
import numpy as np
def get_rays(H, W, focal, c2w):
    i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H), indexing='xy')
    i, j = i.to(c2w.device), j.to(c2w.device)
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

def render_rays(model, rays_o, rays_d, near=2.0, far=6.0, N_samples=64):
    z_vals = torch.linspace(near, far, N_samples).to(rays_o.device)
    z_vals = z_vals.expand(rays_o.shape[0], N_samples) 
    
    mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
    upper = torch.cat([mids, z_vals[...,-1:]], -1)
    lower = torch.cat([z_vals[...,:1], mids], -1)
    t_rand = torch.rand(z_vals.shape).to(rays_o.device)
    z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [Batch, N_samples, 3]
    
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = rays_d[:,None,:].expand_as(pts).reshape(-1, 3)
    
    raw = model(pts_flat, dirs_flat)
    raw = raw.reshape(rays_o.shape[0], N_samples, 4)
    
    rgb = raw[..., :3]
    sigma = raw[..., 3]
    
    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.tensor([1e10], device=device).expand(dists[...,:1].shape)], -1)
    alpha = 1.-torch.exp(-sigma * dists)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    
    rgb_map = torch.sum(weights[...,None] * rgb, -2)
    acc_map = torch.sum(weights, -1)
    
    return rgb_map + (1.-acc_map[...,None])

