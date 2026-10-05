import torch
import torch.nn as nn
import torch.nn.functional as F

class SiamFCLoss(nn.Module):
    def __init__(self, radius=16):
        super().__init__()
        self.radius = radius
        
        coords = 63 + torch.arange(17) * 8
        self.register_buffer("cell_x", coords.view(1, 17))  # cols (1x17)
        self.register_buffer("cell_y", coords.view(17, 1))  # rows (17x1)
        # do this once so u dont recreate in the gt_target

    def create_ground_truth_target(self, shift_x, shift_y):
        # shift_x, shift_y of shape (B,)
        target_x = (127 - shift_x).view(-1, 1, 1, 1).float()
        target_y = (127 - shift_y).view(-1, 1, 1, 1).float()
        
        dist_sq = (self.cell_x - target_x) ** 2 + (self.cell_y - target_y) ** 2 # broadcasts to 17x17 with sq distances
        
        target_map = torch.where(dist_sq <= self.radius**2, 1.0, -1.0)
        
        return target_map # (B, 1, )
        
    def forward(self, pred_heatmaps, shift_x, shift_y):
        # pred_heatmaps is shape (B, 1, 17, 17)
        # shift_x, shift_y are (B,)
        
        targets = self.create_ground_truth_target(shift_x, shift_y)
        
        pos_mask = (targets == 1.0)
        neg_mask = (targets == -1.0)
        
        pos_counts = pos_mask.sum(dim=(1, 2, 3), keepdim=True).float().clamp(min=1.0) # (B, 1, 1, 1) mask with how many instances of class per batch
        neg_counts = neg_mask.sum(dim=(1, 2, 3), keepdim=True).float().clamp(min=1.0)
        
        # every single pixel loss is scaled by that. Positives as well ass negatives apply for 50% of the loss.
        pos_weight_per_img = 0.5 / pos_counts
        neg_weight_per_img = 0.5 / neg_counts
        
        weights = torch.where(targets == 1.0, pos_weight_per_img, neg_weight_per_img)
        
        loss = F.softplus(-targets * pred_heatmaps)
        weighted_loss = loss * weights
        final_loss = weighted_loss.sum(dim=(1, 2, 3)).mean() # sum over dims, mean over batch
        
        return final_loss

        
        