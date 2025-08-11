import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage import img_as_float
import imageShiftOrdering as so

# Load and preprocess image
image_dir = '/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'
#image_dir='/Users/johnwhitfield/Desktop/proper/t:19:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'
#image_dir='/Users/johnwhitfield/Desktop/proper/t:21:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

img = np.array(Image.open(image_dir))

if len(img.shape) == 3:
    img = rgb2gray(img)
image = img_as_float(img)

# Compute maps (assuming so.compute_orientation_map exists)
orientation_map, S_map = so.compute_orientation_map(image, window_radius=7)

xsplit, ysplit = 4, 4
SGrid, DirectorsGrid = so.compute_cell_directors_from_map(orientation_map, S_map, xsplit, ysplit)

# Create a 1x4 subplot layout
plt.figure(figsize=(16, 4))  # Wider figure to accommodate 4 panels

# --- Panel 1: Original ---
ax1 = plt.subplot(141)  # Changed to 141 for 4 panels
ax1.imshow(image, cmap='gray')
ax1.set_title("Original")
ax1.axis('off')

# --- Panel 2: Orientation Map ---
ax2 = plt.subplot(142)  # Changed to 142
im = ax2.imshow(orientation_map, cmap='hsv')
ax2.set_title("Orientation Map")
ax2.axis('off')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# --- Panel 3: Order Parameter (Pixel-wise) ---
ax3 = plt.subplot(143)  # Changed to 143
im = ax3.imshow(S_map, cmap='viridis')
ax3.set_title("Order Parameter (S)")
ax3.axis('off')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

# --- Panel 4: Cell Directors Overlay ---
ax4 = plt.subplot(144)  # Changed to 144

# Display the original image as background
ax4.imshow(image, cmap='gray')

# Overlay the cell directors (using your function)
ysplit, xsplit = SGrid.shape
cell_height = image.shape[0] // ysplit
cell_width = image.shape[1] // xsplit

# Draw cell boundaries (optional, same as in plot_cell_directors)
for y in range(1, ysplit):
    ax4.axhline(y*cell_height, color='white', alpha=0.5, linestyle='--')
for x in range(1, xsplit):
    ax4.axvline(x*cell_width, color='white', alpha=0.5, linestyle='--')

# Plot directors (same logic as in plot_cell_directors)
scale = 0.4 * min(cell_height, cell_width)

for y in range(ysplit):
    for x in range(xsplit):
        if SGrid[y,x] > 0.1:  # Only plot significant order
            angle = DirectorsGrid[y,x]
            center_x = (x + 0.5) * cell_width
            center_y = (y + 0.5) * cell_height
            dx = scale * SGrid[y,x] * np.cos(angle)
            dy = scale * SGrid[y,x] * np.sin(angle)
            
            ax4.arrow(center_x - dx/2, center_y - dy/2, dx, dy,
                     head_width=scale*0.15, head_length=scale*0.2,
                     fc='red', ec='red', width=scale*0.03)

ax4.set_title("Cell Directors (Averaged)")
ax4.axis('off')

plt.tight_layout()
plt.show()