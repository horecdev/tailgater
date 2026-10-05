import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
from pathlib import Path

class SiamFCDataset(Dataset):
    def __init__(self, processed_dir: Path):
        self.videos = []
        
        for vid_dir in processed_dir.iterdir():
            if vid_dir.is_dir():
                frames = sorted(list(vid_dir.glob("*.jpg")))
                if len(frames) > 2:
                    self.videos.append(frames)
                    
    def __len__(self):
        return len(self.videos)
    
    def __getitem__(self, idx):
        video_frames = self.videos[idx]
        num_frames = len(video_frames)
        
        idx_A = random.randint(0, num_frames - 1)
        idx_B = random.randint(0, num_frames - 1)
        while idx_B == idx_A and num_frames > 1:
            idx_B = random.randint(0, num_frames - 1)
        
        img_A : np.ndarray = cv2.imread(str(video_frames[idx_A])) # type: ignore
        img_B : np.ndarray = cv2.imread(str(video_frames[idx_B])) # type: ignore
        
        exemplar = img_A[192:319, 192:319] # grab the inner 127x127 (center object)
        
        shift_x = random.randint(-64, 64)
        shift_y = random.randint(-64, 64)
        
        # 255x255 center in 511x511 is 128 -> 383
        start_x = 128 + shift_x
        start_y = 128 + shift_y
        end_x = 383 + shift_x
        end_y = 383 + shift_y
        
        search = img_B[start_y:end_y, start_x:end_x]
        
        exemplar_tensor = torch.from_numpy(exemplar).permute(2, 0, 1).float() / 255.0
        search_tensor = torch.from_numpy(search).permute(2, 0, 1).float() / 255.0
        
        return exemplar_tensor, search_tensor, shift_x, shift_y
            