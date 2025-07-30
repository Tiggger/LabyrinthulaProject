from PIL import Image
import numpy as np
import re
import os
import copy

# Load image
inputDir='/Users/johnwhitfield/Desktop/input/'

filePrefix='2025-07-14_10x_BF_tile_1_MMStack_1'

#2025-07-14_10x_BF_tile_1_MMStack_1-Pos000_000.ome

rows, cols = 1, 4

# Extract the grid indices from the filename
def get_grid_indices(filename):
    match = re.search(r'Pos(\d{3})_(\d{3})', filename)
    if match:
        x_index = int(match.group(1))
        y_index = int(match.group(2))
        return x_index, y_index
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    
# Load all image filenames
image_files = [f for f in os.listdir(inputDir) if f.endswith('.tif')]


pc=[]

print(image_files)

for file in image_files[::-1]: #necessary to reverse list
    img=Image.open(inputDir+file)
    pixels=np.array(img)

    count0=np.sum(pixels==0)
    count1=np.sum(pixels==1)

    pc.append(float(count0/(count0+count1)))

print(pc)



#img = Image.open('/Users/johnwhitfield/Desktop/2025-07-14_10x_BF_tile_1_MMStack_1-Pos000_000.ome.tif')
#pixels = np.array(img)


#cell is 0, empty space is 1 (backwards intuition)
#count_0 = np.sum(pixels == 0)  # Count pixels with value 0
#count_1 = np.sum(pixels == 1)

#print(count_0/(count_0+count_1))


