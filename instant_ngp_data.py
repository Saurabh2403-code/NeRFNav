import json
import imageio.v2 as imageio
import cv2
import numpy as np
import os
import torch
device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Define the path to the unzipped folder
BASE_DIR = '/content/lego/lego' 

# Function to load Blender (JSON) formatted data
def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f'transforms_{s}.json'), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    
    # Load Train, Val, and Test data
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        
        # skip defines how many frames we skip (to save memory/time)
        skip = testskip if s == 'test' else 1
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.imread(fname))
            poses.append(np.array(frame['transform_matrix']))

        # Normalize images to [0, 1]
        imgs = (np.array(imgs) / 255.).astype(np.float32) 
        poses = np.array(poses).astype(np.float32)
        
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    # Combine all splits
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    # Calculate Focal Length from Field of View (camera_angle_x)
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    return imgs, poses, H, W, focal