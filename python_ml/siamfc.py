import torch
import torch.nn as nn
import torch.nn.functional as F

# premise of SiamFC: 
# you have exemplar (z) and search (x). You use the same convnet on both -> map them to the same feature space.
# For exmaple, if feature 105 means "yellow texture", and its in both images, both' 105 fires ON

# Why use exemplar as the filter for search?
# Convolution is literally sliding dot product. If activations for channel 105 is high in both, output is high. 
# If low in one, high in the other, output is low (no match).
# It relies on the heuristic that presence = numerically bigger number

# What is logistic loss?
# Our results are 17x17 with the correct answer marked by 1, wrong by -1
# Loss = log(1 + e^(-y_truth * logit))
# If y_truth = -1 and logit = 5.0, loss = log(1 + e^5) = high loss
# If y_truth = 1 and logit = -5.0, loss = log(1 + e^5) = high loss
# If y_truth = 1 and logits = 5.0, loss = log(1 + e^(-5)) = low loss (close to 0)

# BatchNorm2d normalizes each feature channel across all batches. Turns it all into z-scores (normal distrib)

# Why ReLU inplace=true
# To use that on any single tensor:
# 1. in-place layer must compute its gradients using its output, not input (because it overwrites the inputs)
# 2. previous layer (batchnorm) must not rely on its output to compute its own backward pass (because it was overwritten)

class SiamFC(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=2),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, stride=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384, 256, kernel_size=3, stride=1)
        )
        # total of like 3.7M params
        
        # original explodes after cross-correlation so initialize to tiny value (0.001) and set bias to 0 so it acts as simple scale
        self.adjust = nn.Conv2d(1, 1, kernel_size=1)
        self.adjust.weight.data.fill_(1e-3)
        self.adjust.bias.data.zero_()
        
    def forward(self, x, z):
        # x is (B, 3, 255, 255)
        # z is (B, 3, 127, 127)

        x_feat = self.backbone(x)
        z_feat = self.backbone(z)
        
        # x is (B, 256, 22, 22)
        # z is (B, 256, 6, 6)
        
        B = x.shape[0]
        x_reshaped = x_feat.view(1, B * 256, 22, 22)
        # groups means: split the input channel into g distinct groups, split the filters into g distinct groups, and only convolve Group 1 x Filter 1, Group 2 x Filter 2
        # its a smart workaround. cuDNN expects static weights for every image (z) but z differs. By default it would for each X, produce 8 heatmaps out of 8 Z's.
        out = F.conv2d(x_reshaped, z_feat, groups=B)
        heatmap = out.view(B, 1, 17, 17)
        
        return self.adjust(heatmap)