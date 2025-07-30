import os
from PIL import Image, ImageSequence
import re
import copy
import numpy as np

# Set directory containing the tiff files
image_dir = '/Users/johnwhitfield/Desktop/input'
filename_prefix = '2025-07-14_10x_BF_tile_1'
output_dir = '/Users/johnwhitfield/Desktop/output'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Set grid dimensions (rows and columns)
rows, cols = 4, 1

# Define overlap parameter (fraction of image size)
overlap = 0.1  # 10% overlap

# Define the number of timepoints to process
max_timepoints = 1  # Set to desired number of timepoints

# Function to calculate percentage of 0s (cells)
def calculate_percentage(image):
    pixels = np.array(image)
    count0 = np.sum(pixels == 0)  # Cells
    count1 = np.sum(pixels == 1)  # Background
    total = count0 + count1
    return (count0 / total) * 100 if total > 0 else 0

# Extract grid indices from filename
def get_grid_indices(filename):
    match = re.search(r'Pos(\d{3})_(\d{3})', filename)
    if match:
        x_index = int(match.group(1))
        y_index = int(match.group(2))
        return x_index, y_index
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")

# Load all image filenames
image_files = [f for f in os.listdir(image_dir) if f.endswith('.tif')]

# Initialize dictionary to store image stacks with grid positions
images_grid = {}

# Load each image stack and map to its grid position
for file in image_files:
    x_index, y_index = get_grid_indices(file)
    y_index = rows - 1 - y_index  # Adjust for inverted y-axis

    image_path = os.path.join(image_dir, file)
    image_stack = Image.open(image_path)

    # Store frames in dictionary
    images_grid[(x_index, y_index)] = [copy.deepcopy(frame) for frame in ImageSequence.Iterator(image_stack)]

# Determine number of timepoints from one stack
num_timepoints = len(next(iter(images_grid.values())))

# Limit timepoints for debugging
timepoints_to_process = min(max_timepoints, num_timepoints)

# Get tile dimensions (assuming all same size)
first_image = next(iter(images_grid.values()))[0]
tile_width, tile_height = first_image.size

# Calculate step size with overlap
x_step = int(tile_width * (1 - overlap))
y_step = int(tile_height * (1 - overlap))

# Calculate canvas size
stitched_width = x_step * (cols - 1) + tile_width
stitched_height = y_step * (rows - 1) + tile_height

# Process each timepoint
for t in range(timepoints_to_process):
    print(f"\nProcessing time point {t + 1}/{timepoints_to_process}...")
    
    # Create empty canvas for stitching
    stitched_image = Image.new("I", (stitched_width, stitched_height))
    
    # Paste each frame at time t into correct position
    for (x_index, y_index), image_stack in images_grid.items():
        reversed_x_index = cols - 1 - x_index  # Reverse x for correct order
        
        try:
            frame = image_stack[t]  # Get frame at time t
        except IndexError:
            print(f"Warning: Time point {t} missing for tile ({x_index}, {y_index})")
            continue

        # Calculate and print % of 0s (cells)
        percentage = calculate_percentage(frame)
        print(f"Tile (X={x_index}, Y={y_index}): {percentage:.2f}% cells")

        # Calculate position with overlap
        x_pos = reversed_x_index * x_step
        y_pos = y_index * y_step
        
        # Paste the frame into the stitched image
        stitched_image.paste(frame, (x_pos, y_pos))
    
    # Save the stitched image
    output_filename = os.path.join(output_dir, f'{filename_prefix}_timepoint_{t}.tif')
    stitched_image.save(output_filename, compression="tiff_deflate")
    print(f"Saved stitched image: {output_filename}")