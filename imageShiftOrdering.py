import numpy as np
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float
from skimage.color import rgb2gray

def splitIntoCells(img, xsplit, ysplit):
    """Split image into a grid of cells."""
    rows, cols = img.shape[:2]
    cellHeight = rows // ysplit
    cellWidth = cols // xsplit
    cells = []
    
    for y in range(ysplit):
        yStart = y * cellHeight
        yEnd = (y + 1) * cellHeight if y != ysplit - 1 else rows
        rowCells = []
        
        for x in range(xsplit):
            xStart = x * cellWidth
            xEnd = (x + 1) * cellWidth if x != xsplit - 1 else cols
            cell = img[yStart:yEnd, xStart:xEnd]
            rowCells.append(cell)
        cells.append(rowCells)
    return cells

def compute_cell_director(cell):
    """Compute S and director for a single cell."""
    if len(cell.shape) == 3:
        cell = rgb2gray(cell)
    cell = img_as_float(cell)
    
    # Compute orientation field (simplified for single cell)
    angles = np.linspace(0, np.pi, 8, endpoint=False)
    max_corr, best_angle = -1, 0
    
    for theta in angles:
        dx, dy = np.cos(theta), np.sin(theta)
        shifted = shift(cell, (dy, dx), mode='reflect')
        corr = np.sum(cell * shifted) / np.sum(cell**2) if np.sum(cell**2) > 0 else 0
        if corr > max_corr:
            max_corr, best_angle = corr, theta
    
    # Compute Q-tensor
    cos2 = np.cos(2 * best_angle) * max_corr
    sin2 = np.sin(2 * best_angle) * max_corr
    Q_xx, Q_xy = cos2, sin2
    S = np.sqrt(Q_xx**2 + Q_xy**2)  # Simplified eigenvalue
    director = 0.5 * np.arctan2(Q_xy, Q_xx)
    
    return S, director

def drawCellBoundaries(image, xsplit, ysplit, lineValue=1.0, thickness=1):
    """Draw grid boundaries on the image."""
    img = image.copy()
    height, width = img.shape
    x_coords = [i * (width // xsplit) for i in range(1, xsplit)]
    y_coords = [i * (height // ysplit) for i in range(1, ysplit)]
    
    for x in x_coords:
        start, end = max(0, x - thickness//2), min(width, x + thickness//2 + thickness%2)
        img[:, start:end] = lineValue
    for y in y_coords:
        start, end = max(0, y - thickness//2), min(height, y + thickness//2 + thickness%2)
        img[start:end, :] = lineValue
    return img


import numpy as np
from scipy.ndimage import shift
from skimage import img_as_float
from tqdm import tqdm  # For progress bars

def compute_orientation_map(image, shift_distance=1, window_radius=5, num_angles=8):
    """
    Computes pixel-wise orientation and order parameter maps using local shift-correlation.
    
    Args:
        image: 2D grayscale image (normalized to [0, 1])
        shift_distance: Pixel shift distance (default: 1)
        window_radius: Radius around each pixel for local analysis (default: 5 → 11x11 windows)
        num_angles: Number of test angles (default: 8)
        
    Returns:
        orientation_map: 2D array of angles (radians) same size as input
        S_map: 2D array of order parameters [0,1] same size as input
    """
    # Initialize outputs
    h, w = image.shape
    orientation_map = np.zeros((h, w))
    S_map = np.zeros((h, w))
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    
    # Pad image to handle boundaries
    padded_img = np.pad(image, window_radius, mode='reflect')
    
    for y in tqdm(range(h), desc="Computing orientation map"):
        #print(y, 'ypos')
        for x in range(w):
            # Extract local window (add padding offset)
            window = padded_img[y:y+2*window_radius+1, x:x+2*window_radius+1]
            
            max_corr, best_angle = -1, 0
            for theta in angles:
                dx = shift_distance * np.cos(theta)
                dy = shift_distance * np.sin(theta)
                
                # Shift and correlate
                shifted = shift(window, (dy, dx), mode='reflect')
                corr = np.sum(window * shifted) / (np.sum(window**2) + 1e-6)
                
                if corr > max_corr:
                    max_corr, best_angle = corr, theta
            
            # Compute Q-tensor components
            cos2 = np.cos(2*best_angle) * max_corr
            sin2 = np.sin(2*best_angle) * max_corr
            
            orientation_map[y, x] = 0.5 * np.arctan2(sin2, cos2)
            S_map[y, x] = np.sqrt(cos2**2 + sin2**2)
    
    return orientation_map, S_map


# Main workflow
"""
image_path = "/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg"
image = io.imread(image_path)
if len(image.shape) == 3:
    image = rgb2gray(image)
image = img_as_float(image)

# Split into cells (e.g., 7x7 grid)
xsplit, ysplit = 7, 7
cells = splitIntoCells(image, xsplit, ysplit)

# Compute S and directors for each cell
S_grid = np.zeros((ysplit, xsplit))
directors_grid = np.zeros((ysplit, xsplit))
for y in range(ysplit):
    for x in range(xsplit):
        S_grid[y, x], directors_grid[y, x] = compute_cell_director(cells[y][x])

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image, cmap='gray')

# Draw cell boundaries
image_with_grid = drawCellBoundaries(image, xsplit, ysplit, lineValue=np.max(image))
ax.imshow(image_with_grid, cmap='gray', alpha=0.3)

# Plot directors (scaled by S)
cell_height = image.shape[0] // ysplit
cell_width = image.shape[1] // xsplit
scale = 0.5 * min(cell_height, cell_width)  # Scale factor for arrow lengths

for y in range(ysplit):
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
"""