# Function to radially dilate endpoints

import numpy as np
import math
from scipy.ndimage import binary_dilation

# Intial function that just isotropically expands the endpoints

def dilate_endpoints_isotropic(endpoints, radius):
    # Create a circular structuring element for dilation
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    circular_mask = x**2 + y**2 <= radius**2

    # Perform binary dilation using the circular mask
    dilated_endpoints = binary_dilation(endpoints, structure=circular_mask)

    return dilated_endpoints


def bresenham_line(x0, y0, x1, y1):
    """Generate points along a line using Bresenham's algorithm."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    
    return points


def create_filament_mask(radius, angle):
    """Create a binary mask with a filament extending from the center."""
    size = int(np.ceil(radius) // 2 * 2 + 1)
    mask = np.zeros((size, size), dtype=bool)
    x0, y0 = size // 2, size // 2  # Start from the center
    x1 = int(x0 + radius * math.cos(angle))
    y1 = int(y0 - radius * math.sin(angle))  # Negative sin to match image coordinates
    
    line_points = bresenham_line(x0, y0, x1, y1)
    for x, y in line_points:
        if 0 <= x < size and 0 <= y < size:
            mask[y, x] = True
    
    return mask


def iterate_dilation(endpoint_positions,radius, norm_phi, binary_image):
    """
    Iterate the process of extension for a given radius, given endpoint positions and the orientation map
    
    """


    from Functions.ProcessSkeleton import get_neighbour_positions,select_angle
    from Functions.Dilation import create_filament_mask

    # Create a copy of the binary image to apply the dilations
    dilated_skeleton = np.copy(binary_image)

    # Process each endpoint
    for endpoint in endpoint_positions:
        # Get neighbours and their positions
        neighbours = get_neighbour_positions(endpoint, binary_image)
        
        for pos, coord in neighbours:
            
            # Get the corresponding value from norm_phi at this coordinate
            norm_phi_value = norm_phi[coord[0], coord[1]]  # Assuming norm_phi is a 2D numpy array
            
            # Pass the endpoint ID (pos) and norm_phi value to select_angle
            angle = select_angle(pos, norm_phi_value)
            
            dilated_endpoint = create_filament_mask(radius =  radius, angle = angle)

            # Get half the mask size (since the mask dimensions are always odd, this ensures proper centering)
            mask_half = (dilated_endpoint.shape[0] // 2, dilated_endpoint.shape[1] // 2)

            # Compute the top-left corner for mask placement
            top_left = (coord[0] - mask_half[0], coord[1] - mask_half[1])

            # Compute valid placement boundaries within the skeleton
            x_start, x_end = max(0, top_left[0]), min(dilated_skeleton.shape[0], top_left[0] + dilated_endpoint.shape[0])
            y_start, y_end = max(0, top_left[1]), min(dilated_skeleton.shape[1], top_left[1] + dilated_endpoint.shape[1])

            # Compute corresponding mask slice to ensure correct alignment
            mask_x_start, mask_x_end = x_start - top_left[0], x_end - top_left[0]
            mask_y_start, mask_y_end = y_start - top_left[1], y_end - top_left[1]

            # Apply the mask to the dilated skeleton
            dilated_skeleton[x_start:x_end, y_start:y_end] |= dilated_endpoint[mask_x_start:mask_x_end, mask_y_start:mask_y_end]

    return dilated_skeleton


def remove_unconnected_extensions(skeleton, new_endpoints, new_contributions):
    """
    Remove filament extensions that have not made a new connection.
    
    Parameters:
    - skeleton: np.array, the binary skeleton after dilation.
    - new_endpoints: np.array, binary mask of endpoints after dilation.
    - new_contributions: np.array, binary mask of newly added pixels (red pixels).
    
    Returns:
    - np.array, updated skeleton with unconnected extensions removed.
    """
    updated_skeleton = np.copy(skeleton)
    
    for endpoint in np.argwhere(new_endpoints):
        x, y = endpoint
        if new_contributions[x, y]:  # Ensure we only process newly added pixels
            to_remove = set()
            stack = [(x, y)]

            while stack:
                cx, cy = stack.pop()
                if new_contributions[cx, cy]:  # If it's a newly added pixel, mark for removal
                    to_remove.add((cx, cy))

                    # Get 8-connected neighbors
                    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < skeleton.shape[0]) and (0 <= ny < skeleton.shape[1]):  # Bounds check
                            if (nx, ny) not in to_remove and new_contributions[nx, ny]:
                                stack.append((nx, ny))

            # Remove the entire tracked extension
            for px, py in to_remove:
                updated_skeleton[px, py] = 0

    return updated_skeleton

import numpy as np
from skimage.morphology import skeletonize

def count_neighbors(x, y, skeleton):
    """
    Count the number of 8-connected neighbors of a given pixel.
    """
    neighbors = [(x-1, y-1), (x-1, y), (x-1, y+1),
                 (x, y-1),         (x, y+1),
                 (x+1, y-1), (x+1, y), (x+1, y+1)]
    
    count = 0
    for nx, ny in neighbors:
        if 0 <= nx < skeleton.shape[0] and 0 <= ny < skeleton.shape[1]:
            if skeleton[nx, ny]:
                count += 1
    
    return count

def remove_unconnected_extensions_new(skeleton, new_endpoints, new_contributions):
    """
    Remove filament extensions that have not made a new connection, stopping deletion at branch points.
    
    Parameters:
    - skeleton: np.array, the binary skeleton after dilation.
    - new_endpoints: np.array, binary mask of endpoints after dilation.
    - new_contributions: np.array, binary mask of newly added pixels (red pixels).
    
    Returns:
    - np.array, updated skeleton with unconnected extensions removed.
    """
    updated_skeleton = np.copy(skeleton)
    
    for endpoint in np.argwhere(new_endpoints):
        x, y = endpoint
        if new_contributions[x, y]:  # Ensure we only process newly added pixels
            to_remove = set()
            stack = [(x, y)]

            while stack:
                cx, cy = stack.pop()
                
                # Check if this pixel has more than 2 neighbors (branch point)
                if count_neighbors(cx, cy, updated_skeleton) > 2:
                    break  # Stop removal here
                
                if new_contributions[cx, cy]:  # If it's a newly added pixel, mark for removal
                    to_remove.add((cx, cy))

                    # Get 8-connected neighbors
                    for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if (0 <= nx < skeleton.shape[0]) and (0 <= ny < skeleton.shape[1]):  # Bounds check
                            if (nx, ny) not in to_remove and new_contributions[nx, ny]:
                                stack.append((nx, ny))

            # Remove only the tracked extension up to a valid connection point
            for px, py in to_remove:
                updated_skeleton[px, py] = 0

    return updated_skeleton


