import cv2
import torch
import numpy as np
import torch.nn.functional as F
from python_ml.siamfc import SiamFC

mouse_x, mouse_y = 320, 420
target_x, target_y = 0, 0

select_scale = 1.0
target_scale = 1.0

is_tracking = False
new_target_requested = False

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y, target_x, target_y, select_scale, target_scale, is_tracking, new_target_requested
    
    mouse_x, mouse_y = x, y
    
    if event == cv2.EVENT_LBUTTONDOWN:
        target_x, target_y = x, y
        target_scale = select_scale
        is_tracking = True
        new_target_requested = True
        
    elif event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0: # zoom out
            select_scale += 0.05
        else: # zoom in
            select_scale -= 0.05
        select_scale = max(0.2, min(select_scale, 5.0))
        
def get_scaled_crop(frame, c_x, c_y, base_size, scale):
    # crops a region from the frame and pads with gray if out of bounds, then resizes back to base_size for SiamFC
    real_size = int(base_size * scale)
    half = real_size // 2
    
    x_1, y_1 = int(c_x) - half, int(c_y) - half
    x_2, y_2 = int(c_x) + half, int(c_y) + half
    
    pad_left = max(0, -x_1)
    pad_top = max(0, -y_1)
    pad_right = max(0, x_2 - frame.shape[1])
    pad_bottom = max(0, y_2 - frame.shape[0])
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        frame = cv2.copyMakeBorder(frame, pad_left, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[128, 128, 128])
        # if x_1 is 20 px to the left (coords -20px) it gets added back the 20px. 
        x_1 += pad_left
        x_2 += pad_left
        y_1 += pad_top
        y_2 += pad_top
        
    crop = frame[y_1:y_2, x_1:x_2]
    
    # resize to base_size (cuz crop itself is scaled)
    resized = cv2.resize(crop, (base_size, base_size), interpolation=cv2.INTER_LINEAR)
    
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor, resized
    
        
def main():
    global new_target_requested, target_x, target_y, target_scale, select_scale, is_tracking
    
    # set up the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiamFC().to(device)
    model.load_state_dict(torch.load("train/siamfc_epoch_61.pth", map_location=device, weights_only=True))
    model.eval()
    exemplar_tensor = None  # To store our locked target
    exemplar_img = None
    
    # without this flag it lags as hell
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cv2.namedWindow("Goated Tracker", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Goated Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setMouseCallback("Goated Tracker", mouse_callback)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # copy the frame we will draw on
        display_frame = frame.copy()
        
        select_half = int((127 * select_scale) / 2) # half the green square side
        # draw the crosshair
        cv2.drawMarker(display_frame, (mouse_x, mouse_y), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)
        # draw the rectangle crosshair around
        cv2.rectangle(display_frame, (mouse_x - select_half, mouse_y - select_half), (mouse_x + select_half, mouse_y + select_half), (0, 255, 0), 1)
        
        if is_tracking:
            target_half = int((127 * target_scale) / 2)
            cv2.rectangle(display_frame, (int(target_x) - target_half, int(target_y) - target_half), (int(target_x) + target_half, int(target_y) + target_half), (0, 0, 255), 2)
            
            if new_target_requested:
                print(f"Locking new target at: {target_x}, {target_y}")
                
                exemplar_tensor, exemplar_img = get_scaled_crop(frame, target_x, target_y, 127, target_scale)
                exemplar_tensor = exemplar_tensor.to(device)
                new_target_requested = False
                
            else:
                scale_multipliers = [0.95, 1.0, 1.05] # scales to test every iteration
                scale_penalties = [0.97, 1.0, 0.97] # changing the scale = slight penality to the heatmap (model has to be sure)
                
                best_score = -float('inf')
                best_scale_idx = 1
                best_dx_255, best_dy_255 = 0, 0
                
                for idx, (multiplier, penalty) in enumerate(zip(scale_multipliers, scale_penalties)):
                    test_scale = target_scale * multiplier
                    search_tensor, _ = get_scaled_crop(frame, target_x, target_y, 255, test_scale)
                    search_tensor = search_tensor.to(device)
                    
                    heatmap = model(search_tensor, exemplar_tensor).squeeze().cpu().detach().numpy()
                    
                    max_idx = np.argmax(heatmap)
                    # uses modulo and div just like gradcraft to figure out where it is in the shape
                    max_row, max_col = np.unravel_index(max_idx, heatmap.shape)
                    
                    peak_score = heatmap[max_row, max_col] * penalty
                    
                    if peak_score > best_score:
                        best_score = peak_score
                        best_scale_idx = idx
                        # update relative to the center (8, 8 is the center in 17x17)
                        best_dx_255 = (max_col - 8) * 8
                        best_dy_255 = (max_row - 8) * 8
                
                target_scale *= scale_multipliers[best_scale_idx]
                np.clip(target_scale, 0.2, 5.0)

                target_x += best_dx_255 * target_scale
                target_y += best_dy_255 * target_scale
                
                # safety clip
                target_x = np.clip(target_x, 0, frame.shape[1])
                target_y = np.clip(target_y, 0, frame.shape[0])
                
                
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.5
        
        # draw solid black rect 
        cv2.rectangle(display_frame, (10, 10), (240, 80), (0, 0, 0), -1)
        
        # draw text on the rect
        status_text = "TRACKING" if is_tracking else "SELECTING"
        cv2.putText(display_frame, f"STATUS:       {status_text}", (20, 30), font, font_scale, (255, 255, 255), 1)
        cv2.putText(display_frame, f"SELECT SCALE: {select_scale:.2f}x", (20, 50), font, font_scale, (0, 255, 0), 1)
        cv2.putText(display_frame, f"TARGET SCALE: {target_scale:.2f}x", (20, 70), font, font_scale, (0, 0, 255), 1)
        
        # draw exemplar + label
        if is_tracking:
            hud_exemplar = cv2.resize(exemplar_img, (80, 80))  # type: ignore
            
            cv2.putText(display_frame, "EXEMPLAR", (20, 100), font, 0.4, (255, 255, 255), 1)
            display_frame[110:190, 20:100] = hud_exemplar
            cv2.rectangle(display_frame, (20, 110), (100, 190), (255, 255, 255), 1)
        
                
        cv2.imshow("Goated Tracker", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
            
        
    camera.release()
    cv2.destroyAllWindows()
            
if __name__ == "__main__":
    main()