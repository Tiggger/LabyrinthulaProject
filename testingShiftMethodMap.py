import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from skimage import img_as_float
import imageShiftOrdering as so

# Load and preprocess image
image_dir = '/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'
img = np.array(Image.open(image_dir))

if len(img.shape) == 3:
    img = rgb2gray(img)
image = img_as_float(img)

# Compute maps (assuming so.compute_orientation_map exists)
orientation_map, S_map = so.compute_orientation_map(image, window_radius=7)

# Corrected visualization
plt.figure(figsize=(15, 5))

# Panel 1: Original
ax1 = plt.subplot(131)
ax1.imshow(image, cmap='gray')
ax1.set_title("Original")
ax1.axis('off')

# Panel 2: Orientation Map
ax2 = plt.subplot(132)
im = ax2.imshow(orientation_map, cmap='hsv')
ax2.set_title("Orientation Map")
ax2.axis('off')
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

# Panel 3: Order Parameter
ax3 = plt.subplot(133)
im = ax3.imshow(S_map, cmap='viridis')
ax3.set_title("Order Parameter (S)")
ax3.axis('off')
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()