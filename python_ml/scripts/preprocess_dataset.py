import numpy as np
import cv2
import math

from pathlib import Path

videos_path = Path("GOT10k-Raw/got10k/train")
dataset_path = Path("GOT-10k-Processed")

for item in videos_path.iterdir():
    if item.is_dir():
        vid_dir = videos_path / item.name
        target_vid_dir = dataset_path / item.name
        
        print(f"Processing: {vid_dir}")
        
        frames_saved = 0
        
        target_vid_dir.mkdir(exist_ok=True)
        
        with open(vid_dir / "absence.label") as file: # filter for 0
            absence = file.readlines()
        with open(vid_dir / "cover.label") as file: # filter for >= 5
            cover = file.readlines()
        with open(vid_dir / "cut_by_image.label") as file: # filter for 0
            cut = file.readlines()
        with open(vid_dir / "groundtruth.txt") as file:
            groundtruth = file.readlines()
        
        frame_count = sum(1 for file in vid_dir.glob('*.jpg'))
        
        for frame_idx in range(0, frame_count):
            frame_name = f"{(frame_idx + 1):08d}.jpg"
            
            is_absent = True if int(absence[frame_idx].strip()) == 1 else False
            is_covered = True if int(cover[frame_idx].strip()) < 5 else False
            is_cut = True if int(cut[frame_idx].strip()) == 1 else False
            
            if any([is_absent, is_covered, is_cut]):
                print(f"Name: {frame_name}, Absent: {is_absent}, Covered: {is_covered}, Cut: {is_cut}")
                continue
            
            x_min, y_min, W, H = list(map(lambda x: float(x), groundtruth[frame_idx].strip().split(",")))
            
            # now preprocess the frame into a 511x511 .jpg and save into dataset_path
            c_x = x_min + (W / 2)
            c_y = y_min + (H / 2)
            
            p = (W + H) / 4
            area = (W + 2 * p) * (H + 2 * p) # pad by p on both sides
            S_exemplar = math.sqrt(area) # lenght of a square centered at (c_x, c_y)
            
            # if u got a 1x100 pole, then just force the square to fit
            if S_exemplar < max(W, H) * 1.1:
                S_exemplar = max(W, H) * 1.1
                
            S_search = S_exemplar * (511.0 / 127.0) # covers 16x more area
            
            image : np.ndarray = cv2.imread(str(vid_dir / frame_name)) # type:ignore
            
            frame_H, frame_W = image.shape[:2]
            
            # continue of an image falls out of jpg bounds: top left and bottom right must be in bounds
            x_1 = math.floor(c_x - (S_search / 2))
            y_1 = math.floor(c_y - (S_search / 2))
            x_2 = math.ceil(c_x + (S_search / 2))
            y_2 = math.ceil(c_y + (S_search / 2))
            
            pad_left = max(0, -x_1)
            pad_top = max(0, -y_1)
            pad_right = max(0, x_2 - frame_W)
            pad_bottom = max(0, y_2 - frame_H)
            
            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[128, 128, 128]) # gray
                
                x_1 += pad_left
                x_2 += pad_left
                y_1 += pad_top
                y_2 += pad_top
                    
            cropped_image = image[int(y_1):int(y_2), int(x_1):int(x_2)] 
            resized_image : np.ndarray = cv2.resize(cropped_image, (511, 511), interpolation=cv2.INTER_LINEAR)
            
            cv2.imwrite(str(target_vid_dir / frame_name), resized_image)
            
            frames_saved += 1
            
        print(f"Saved {frames_saved} frames.")
            
            
        