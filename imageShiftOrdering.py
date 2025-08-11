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


import masterClass as mc
import matplotlib.cm as cm

def ColourMap(norm_phi, cmap_name='gist_rainbow'):
    """
    from the normalised function (angle values mapped to occupy range 0-1, apply the rgb colour map)
    """
    # Apply chosen color map
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm_phi)[..., :3]

    return rgb

def ApplyMask(rgb, img, rgb_id = True, threshold=0.5):

    print(rgb.shape, 'rgb shape')
    print(img.shape, 'img shape')

    #handling if image is floating point value instead
    if img.max() > 1.0:  # Assume 8-bit image
        binary_mask = (img > 127).astype(float)
    else:  # Assume float image
        binary_mask = (img > threshold).astype(float)

    mask_rgb = binary_mask

    if rgb_id:
        # Expand the binary mask to have 3 channels (for RGB)
        mask_rgb = np.stack([binary_mask]*3, axis=-1)  # Shape will be (height, width, 3)

    # Apply the mask to the RGB color map
    masked_rgb = rgb * mask_rgb  # Keeps original color where mask is 1, sets to black where mask is 0

    return masked_rgb

import numpy as np
import matplotlib.pyplot as plt

class TempAnalysis:
    def __init__(self, cell_mask, orientation_map, S_map):
        self.skeletonImage = cell_mask.astype(int)
        self.processed_png = self.skeletonImage
        self.phi_rotated = orientation_map
        self.S_map = S_map

    def calculateOrientationCorrelation(self, coarsening=0):
        yx_indices = np.argwhere(self.skeletonImage > 0)
        if coarsening > 0:
            yx_indices = yx_indices[::(coarsening + 1)]

        num_points = len(yx_indices)
        if num_points < 2:
            raise ValueError("Not enough masked points to compute correlation.")

        self.distance_list = []
        self.correlation_list = []

        mask_indices = np.argwhere(self.processed_png > 0)
        for y1, x1 in yx_indices:
            angle1 = self.phi_rotated[y1, x1]
            v1 = np.array([np.cos(angle1), np.sin(angle1)])

            for y2, x2 in mask_indices:
                if (y1, x1) == (y2, x2):
                    self.distance_list.append(0)
                    self.correlation_list.append(1)
                    continue

                angle2 = self.phi_rotated[y2, x2]
                v2 = np.array([np.cos(angle2), np.sin(angle2)])
                dot_product = np.abs(np.dot(v1, v2))
                distance = np.linalg.norm([y2 - y1, x2 - x1])
                self.distance_list.append(distance)
                self.correlation_list.append(dot_product**2)

        return self.distance_list, self.correlation_list

    def binIt(self, bin_size):
        max_distance = max(self.distance_list)
        bins = np.arange(0, max_distance + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size / 2

        correlation_avg = np.zeros(len(bin_centers))
        counts = np.zeros(len(bin_centers))
        std_dev = np.zeros(len(bin_centers))
        bin_values = [[] for _ in range(len(bin_centers))]

        for d, c in zip(self.distance_list, self.correlation_list):
            bin_index = int(d // bin_size)
            if bin_index < len(correlation_avg):
                bin_values[bin_index].append(c)
                correlation_avg[bin_index] += c
                counts[bin_index] += 1

        for i in range(len(bin_centers)):
            if counts[i] > 0:
                correlation_avg[i] /= counts[i]
                std_dev[i] = np.std(bin_values[i]) if len(bin_values[i]) > 1 else 0

        std_err = np.zeros(len(bin_centers))
        std_err[counts > 0] = std_dev[counts > 0] / np.sqrt(counts[counts > 0])
        correlation_avg[counts == 0] = np.nan
        std_err[counts == 0] = np.nan

        return bin_centers, correlation_avg, counts, std_err

    def calculateCorrelationAvgNematic(self, correlation_avg):
        return (correlation_avg - 0.5) / 0.5


def createInteractiveOrderingHeatMap(image, S_grid, directors_grid, cells, orientationMap, sMap, magnification=20):
    """Interactive plot showing cell directors; click to analyse correlation."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(image, cmap='gray')

    ysplit, xsplit = S_grid.shape
    cell_height = image.shape[0] // ysplit
    cell_width = image.shape[1] // xsplit

    # Draw grid lines
    for y in range(1, ysplit):
        ax.axhline(y * cell_height, color='white', alpha=0.5, linestyle='--')
    for x in range(1, xsplit):
        ax.axvline(x * cell_width, color='white', alpha=0.5, linestyle='--')

    # Plot directors
    scale = 0.4 * min(cell_height, cell_width)
    for y in range(ysplit):
        for x in range(xsplit):
            if S_grid[y, x] > 0.1:
                angle = directors_grid[y, x]
                center_x = (x + 0.5) * cell_width
                center_y = (y + 0.5) * cell_height
                dx = scale * S_grid[y, x] * np.cos(angle)
                dy = scale * S_grid[y, x] * np.sin(angle)
                ax.arrow(center_x - dx / 2, center_y - dy / 2, dx, dy,
                         head_width=scale * 0.15, head_length=scale * 0.2,
                         fc='red', ec='red', width=scale * 0.03)

    plt.title("Click on any cell to analyse (Image Shift Method)")

    def onclick(event):
        if event.inaxes != ax:
            return

        # Identify clicked cell
        x_click, y_click = int(event.xdata), int(event.ydata)
        cell_x = x_click // cell_width
        cell_y = y_click // cell_height

        if cell_y >= len(cells) or cell_x >= len(cells[0]):
            return

        # Bounds of clicked cell in full image
        y_start = cell_y * cell_height
        y_end = y_start + cell_height
        x_start = cell_x * cell_width
        x_end = x_start + cell_width

        # Get mask for clicked cell
        clicked_cell_mask = cells[cell_y][cell_x]

        # Slice matching region from orientation and S maps
        cell_orient_full = orientationMap[y_start:y_end, x_start:x_end]
        cell_S_full = sMap[y_start:y_end, x_start:x_end]

        # Mask values so only cell pixels remain
        cell_orient = np.where(clicked_cell_mask > 0, cell_orient_full, np.nan)
        cell_S = np.where(clicked_cell_mask > 0, cell_S_full, np.nan)

        S_value = S_grid[cell_y, cell_x]
        director = directors_grid[cell_y, cell_x]

        # Plot cell + correlation analysis
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(clicked_cell_mask, cmap='gray')
        ax1.set_title(f"Cell ({cell_x}, {cell_y})\nS = {S_value:.3f}, Director = {director:.3f} rad")
        ax1.axis('off')

        analysis = TempAnalysis(clicked_cell_mask, cell_orient, cell_S)
        distance_list, correlation_list = analysis.calculateOrientationCorrelation(coarsening=1)
        bin_centers, correlation_avg, counts, std_err = analysis.binIt(bin_size=2)
        correlation_avg_nematic = analysis.calculateCorrelationAvgNematic(correlation_avg)

        pixel_sizes = {20: 0.3236, 10: 0.651, 4: 1.6169}
        pixelSize = pixel_sizes.get(magnification, 1.0)
        bin_centers_microns = bin_centers * pixelSize

        ax2.errorbar(bin_centers_microns, correlation_avg_nematic, yerr=std_err, label='Orientation Correlation')
        ax2.set_xlabel('Distance ($\mu m$)')
        ax2.set_ylabel('Correlation')
        ax2.set_title('Orientation Correlation Function')
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.tight_layout()
    plt.show()




from PIL import Image
from skimage.color import rgb2gray
from skimage import img_as_float

image_dir = '/Users/johnwhitfield/Desktop/proper/t:21:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'
image = np.array(Image.open(image_dir))

if len(image.shape) == 3:
    image = rgb2gray(image)
image = img_as_float(image)

xsplit, ysplit = 5, 5

cells=splitIntoCells(image, xsplit, ysplit)

orientationMap, sMap = compute_orientation_map(image)

#normalising angles
norm_phi = (orientationMap + np.pi/2) / np.pi
#creating colour map
rgb=ColourMap(norm_phi)

masked = ApplyMask(rgb, image)

sGrid, directorsGrid = compute_cell_directors_from_map(orientationMap, sMap, xsplit, ysplit)
#plot_cell_directors(image, sGrid, directorsGrid)
createInteractiveOrderingHeatMap(masked, sGrid, directorsGrid, cells, orientationMap, sMap) #can pass image instead of masked
