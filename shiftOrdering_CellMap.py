import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage import img_as_float
import imageShiftOrdering as so 
import imageProcessingTool as dc

# Main workflow
"""
image_dir = '/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'
img = np.array(Image.open(image_dir))
if len(img.shape) == 3:
    img = rgb2gray(img)
image = img_as_float(img)
"""
#testing on synthetic data - not looking too clever
L, d, N = 200, 10, 10 
image, angleDomain = dc.generate_nematic(L, d, N, mode='domain', domain_size=5)

# 1. Compute full orientation map
orientation_map, S_map = so.compute_orientation_map(image, window_radius=7)

# 2. Analyze at cell level (e.g., 7x7 grid)
xsplit, ysplit = 2, 2
S_grid, directors_grid = so.compute_cell_directors_from_map(orientation_map, S_map, xsplit, ysplit)

plt.imshow(np.degrees(orientation_map))
plt.colorbar()

# 3. Visualize
so.plot_cell_directors(image, S_grid, directors_grid)

# Optional: Compare with direct cell analysis
#cells = so.splitIntoCells(image, xsplit, ysplit)
#S_grid_direct, directors_grid_direct = np.zeros((ysplit,xsplit)), np.zeros((ysplit,xsplit))
#for y in range(ysplit):
#    for x in range(xsplit):
#        S_grid_direct[y,x], directors_grid_direct[y,x] = so.compute_cell_director(cells[y][x])

#so.plot_cell_directors(image, S_grid_direct, directors_grid_direct)