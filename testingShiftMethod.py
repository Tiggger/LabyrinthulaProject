import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

import imageShiftOrdering as so

#image_dir='/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

#img = np.array(Image.open(image_dir))

#creat synthetic data to test on
L, d, N = 200, 10, 10 
imageDomain, angleDomain = dc.generate_nematic(L, d, N, mode='domain', domain_size=5)

orientationField, correlationStrength =so.compute_orientation_field(imageDomain)

S, directors = so.calculate_nematic_tensor(orientationField, correlationStrength)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(imageDomain, cmap='gray')
axes[0].set_title("Original Image")

axes[1].imshow(orientationField, cmap='hsv')
axes[1].set_title("Orientation Field (HSV)")

# Plot directors (quiver plot)
y, x = np.mgrid[:S.shape[0], :S.shape[1]]
axes[2].imshow(S, cmap='viridis')
axes[2].quiver(x[::5, ::5], y[::5, ::5], 
                np.cos(directors[::5, ::5]), np.sin(directors[::5, ::5]), 
                scale=20, color='red')
axes[2].set_title("Nematic Order (S) and Directors")

fig.tight_layout()
plt.show()

