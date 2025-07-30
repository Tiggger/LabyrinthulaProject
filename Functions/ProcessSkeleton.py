# Function to apply the kernel and highlight endpoints

import numpy as np # type: ignore
from scipy.ndimage import convolve # type: ignore


def process_skeleton(image, A=10):
    # Define the kernel
    kernel = np.array([
        [1, 1, 1],
        [1, A, 1],
        [1, 1, 1]
    ])

    # Apply the kernel to the image
    processed_image = convolve(image.astype(float), kernel, mode='constant', cval=0.0)

    # Identify endpoints (value A+1 in the processed matrix)
    endpoints = (processed_image == A + 1)

    # Identify pixels with two neighbors (value A+2)
    two_neighbors = (processed_image == A + 2)

    # Identify pixels with three neighbors (value A+3)
    three_neighbors = (processed_image == A + 3)

    # Create a colored image to highlight endpoints and other features
    result_image = np.stack([image] * 3, axis=-1)  # Convert to RGB
    result_image = result_image.astype(float)  # Ensure float for coloring

    # Highlight endpoints in red
    result_image[endpoints, 0] = 1.0  # Red channel
    result_image[endpoints, 1] = 0.0  # Green channel
    result_image[endpoints, 2] = 0.0  # Blue channel

    # Highlight pixels with two neighbors in blue
    result_image[two_neighbors, 0] = 0.0  # Red channel
    result_image[two_neighbors, 1] = 0.0  # Green channel
    result_image[two_neighbors, 2] = 1.0  # Blue channel

    # Highlight pixels with three neighbors in green
    result_image[three_neighbors, 0] = 0.0  # Red channel
    result_image[three_neighbors, 1] = 1.0  # Green channel
    result_image[three_neighbors, 2] = 0.0  # Blue channel

    # Set non-highlighted pixels to their original grayscale values
    others = ~(endpoints | two_neighbors | three_neighbors)
    result_image[others, 0] = image[others]
    result_image[others, 1] = image[others]
    result_image[others, 2] = image[others]

    return processed_image, result_image, endpoints




# Function to determine neighbour positions for a given endpoint
def get_neighbour_positions(endpoint, skeleton):
    row, col = endpoint
    neighbours = []
    positions = {
        (1, -1): 1,  # Bottom Left
        (0, -1): 2,  # Center Left
        (-1, -1): 3, # Top Left
        (-1, 0): 4,  # Top Center
        (-1, 1): 5,  # Top Right
        (0, 1): 6,   # Center Right
        (1, 1): 7,   # Bottom Right
        (1, 0): 8    # Bottom Center
    }
    for (dr, dc), position in positions.items():
        nr, nc = row + dr, col + dc
        # Check if neighbor is within bounds and part of the skeleton
        if 0 <= nr < skeleton.shape[0] and 0 <= nc < skeleton.shape[1]:
            if skeleton[nr, nc] == 1:
                neighbours.append((position, (nr, nc)))  # Position number and coordinates
    return neighbours



# Function to check if an angle lies within a given range or ranges
def is_in_range(angle, ranges):
    """
    Check if an angle is within the specified range(s).

    Args:
        angle (float): The angle to check (in radians).
        ranges (tuple or list of tuples): The range(s) to check against.

    Returns:
        bool: True if the angle lies within any of the ranges.
    """
    # Ensure ranges is always a tuple of tuples for consistency
    if not isinstance(ranges[0], tuple):
        ranges = (ranges,)
    
    for angle_min, angle_max in ranges:
        if angle_min < angle_max:  # Normal range
            if angle_min <= angle <= angle_max:
                return True
        else:  # Handle wrap-around range (e.g., -pi to -pi/4 OR 3pi/4 to pi)
            if angle >= angle_min or angle <= angle_max:
                return True
    return False




# Function to determine the correct angle for dilation
def select_angle(neighbour_location, norm_phi):
    """
    Select the correct angle based on the neighbour location and angle ranges.

    Args:
        neighbor_location (int): The neighbour location (1-8).
        angle1 (float): The first possible angle.
        angle2 (float): The second possible angle.

    Returns:
        float: The selected angle.
    """
    angle_ranges = {
    1: (-np.pi / 4, 3 * np.pi / 4),                     # Bottom Left
    2: (-np.pi / 2, np.pi / 2),                         # Center Left
    3: (-3 * np.pi / 4, np.pi / 4),                     # Top Left
    4: (-np.pi, 0),                                     # Top Center
    5: ((-np.pi, -np.pi / 4), (3 * np.pi / 4, np.pi)),  # Top Right
    6: ((-np.pi, -np.pi / 2), (np.pi / 2, np.pi)),      # Center Right
    7: ((-np.pi, -3 * np.pi / 4), (np.pi / 4, np.pi)),  # Bottom Right
    8: (0, np.pi),                                      # Bottom Center
    }   

    ranges = angle_ranges[neighbour_location]

    # Generate two possible angles from S
    angle1 = ((norm_phi-0.5) * np.pi)   # Map S from [0, 1] to [-pi, pi]
    angle2 = angle1 + np.pi       # The opposite angle

    # Wrap angle2 to [-pi, pi]
    angle2 = angle2 - 2 * np.pi if angle2 > np.pi else angle2

    # Check which angle falls within the specified range
    if is_in_range(angle1, ranges):
        return angle1
    elif is_in_range(angle2, ranges):
        return angle2
    else:
        raise ValueError("Neither angle satisfies the neighbour's criteria. Check inputs.")
    


# Now redundant because of skimage morphology module (MARK FOR REMOVAL)
def process_binary_image(image, hole_min, segment_min):
    """
    Fills small gaps (4-connectivity) and removes small segments (8-connectivity) in a binary image.

    :param image: Binary image (numpy array, dtype uint8, values 0 or 255)
    :param gap_min: Maximum size of a black gap to be filled
    :param segment_min: Maximum size of an isolated white segment to be removed
    :return: Processed binary image
    """

    from scipy.ndimage import label

    if image is None:
        raise ValueError("Error: Image not loaded. Check the file path!")

    # Convert to boolean (True for white, False for black)
    binary = image > 128

    # ---- GAP FILLING (BLACK REGIONS) ----
    inverse = ~binary  # Logical NOT (black becomes white)

    # Define a 4-connectivity structuring element
    structure_4 = np.array([[0, 1, 0], 
                            [1, 1, 1], 
                            [0, 1, 0]], dtype=np.int8)

    # Label connected components in the inverse image (black regions)
    labeled_gaps, num_gaps = label(inverse, structure=structure_4)

    # Fill small black gaps
    for i in range(1, num_gaps + 1):  # Skip background (0)
        if np.sum(labeled_gaps == i) < hole_min:
            binary[labeled_gaps == i] = True  # Fill small gaps

    # ---- SEGMENT PRUNING (WHITE REGIONS) ----
    # Define an 8-connectivity structuring element
    structure_8 = np.ones((3, 3), dtype=np.int8)

    # Label connected components in the original binary image (white regions)
    labeled_segments, num_segments = label(binary, structure=structure_8)

    # Remove small white segments
    for i in range(1, num_segments + 1):  # Skip background (0)
        if np.sum(labeled_segments == i) < segment_min:
            binary[labeled_segments == i] = False  # Remove small white segments

    return binary.astype(np.uint8) * 255  # Convert back to 0 and 255 (uint8)


def SaveFigure(skeleton, filename, rgb = True):
    from PIL import Image
    
    # Ensure correct shape (remove unnecessary dimensions)
    corrected_shape = np.squeeze(skeleton)

    # **Fix 1: If data is in 0-1 range, scale to 0-255**
    if corrected_shape.max() <= 1.0:
        corrected_shape = (corrected_shape * 255)

    # **Fix 2: Ensure valid uint8 format**
    corrected_format = np.clip(corrected_shape, 0, 255).astype(np.uint8)

    if rgb:
        # Convert to PIL Image
        corrected_format = Image.fromarray(corrected_format, mode="RGB")
    else:
        corrected_format = Image.fromarray(corrected_format, mode="L")


    # Save the image
    corrected_format.save(filename, format="PNG")