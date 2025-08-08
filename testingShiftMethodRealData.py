import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

import imageShiftOrdering as so
from skimage.color import rgb2gray


image_dir='/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

img = np.array(Image.open(image_dir))

if len(img.shape)==3:
    img=rgb2gray(img)

# Split into cells (e.g., 2x2 grid)
xsplit, ysplit = 10, 10
cells = so.splitIntoCells(img, xsplit, ysplit)

# Compute S and directors for each cell
S_grid = np.zeros((ysplit, xsplit))
directors_grid = np.zeros((ysplit, xsplit))
for y in range(ysplit):
    for x in range(xsplit):
        S_grid[y, x], directors_grid[y, x] = so.compute_cell_director(cells[y][x])

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(img, cmap='gray')

# Draw cell boundaries
image_with_grid = so.drawCellBoundaries(img, xsplit, ysplit, lineValue=np.max(img))
ax.imshow(image_with_grid, cmap='gray', alpha=0.3)

# Plot directors (scaled by S)
cell_height = img.shape[0] // ysplit
cell_width = img.shape[1] // xsplit
scale = 0.5 * min(cell_height, cell_width)  #Scale factor for arrow lengths

for y in range(ysplit):
    print(ysplit, 'ysplit')
    for x in range(xsplit):
        S = S_grid[y, x]
        angle = directors_grid[y, x]
        center_x = (x + 0.5) * cell_width
        center_y = (y + 0.5) * cell_height
        dx = scale * S * np.cos(angle)
        dy = scale * S * np.sin(angle)
        
        ax.arrow(center_x - dx/2, center_y - dy/2, dx, dy, 
                 head_width=scale*0.2, head_length=scale*0.3, 
                 fc='red', ec='red', width=scale*0.05)

plt.title("Nematic Directors Scaled by S (Cell-wise)")
plt.show()