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

#orientation map cell ordering

def compute_cell_directors_from_map(orientation_map, S_map, xsplit, ysplit):
    """Calculate average S and director for each cell from orientation map"""
    h, w = orientation_map.shape
    cell_height = h // ysplit
    cell_width = w // xsplit
    
    S_grid = np.zeros((ysplit, xsplit))
    directors_grid = np.zeros((ysplit, xsplit))
    
    for y in range(ysplit):
        for x in range(xsplit):
            y_start = y * cell_height
            y_end = (y+1)*cell_height if y != ysplit-1 else h
            x_start = x * cell_width
            x_end = (x+1)*cell_width if x != xsplit-1 else w
            
            # Extract cell region
            cell_orient = orientation_map[y_start:y_end, x_start:x_end]
            cell_S = S_map[y_start:y_end, x_start:x_end]
            
            # Convert orientations to Q-tensor components
            cos2 = np.cos(2*cell_orient) * cell_S
            sin2 = np.sin(2*cell_orient) * cell_S
            
            # Average over cell
            Q_xx = np.mean(cos2)
            Q_xy = np.mean(sin2)
            
            # Compute S and director
            S_grid[y,x] = np.sqrt(Q_xx**2 + Q_xy**2)
            directors_grid[y,x] = 0.5 * np.arctan2(Q_xy, Q_xx)
    
    return S_grid, directors_grid

def plot_cell_directors(image, S_grid, directors_grid):
    """Plot image with cell directors scaled by S"""
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(image, cmap='gray')
    
    ysplit, xsplit = S_grid.shape
    cell_height = image.shape[0] // ysplit
    cell_width = image.shape[1] // xsplit
    
    # Draw cell boundaries
    for y in range(1, ysplit):
        ax.axhline(y*cell_height, color='white', alpha=0.5, linestyle='--')
    for x in range(1, xsplit):
        ax.axvline(x*cell_width, color='white', alpha=0.5, linestyle='--')
    
    # Plot directors
    scale = 0.4 * min(cell_height, cell_width)
    
    for y in range(ysplit):
        for x in range(xsplit):
            if S_grid[y,x] > 0.1:  # Only plot significant order
                angle = directors_grid[y,x]
                center_x = (x + 0.5) * cell_width
                center_y = (y + 0.5) * cell_height
                dx = scale * S_grid[y,x] * np.cos(angle)
                dy = scale * S_grid[y,x] * np.sin(angle)
                
                ax.arrow(center_x - dx/2, center_y - dy/2, dx, dy,
                         head_width=scale*0.15, head_length=scale*0.2,
                         fc='red', ec='red', width=scale*0.03)
    
    plt.title("Cell Directors Scaled by S")
    plt.show()