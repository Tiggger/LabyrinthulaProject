import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image






image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/2.0%/2025-07-28_BF_20xwithExtender_featureImagesDifferentPlate_exports/2025-07-28_BF_20xwithExtender_featureImagesDifferentPlate_MMStack_Default.ome4.jpg'



img = np.array(Image.open(image_dir))
xsplit = 8
ysplit = 8
cells = dc.splitIntoCells(img, xsplit, ysplit)

kernelSize=6
threshold=170 #120

a=mc.ImageAnalysis(image_dir, None, 4, kernelSize, threshold)

densityCells = dc.splitIntoCells(a.binary_Image, xsplit, ysplit)

binaryDensities=dc.calculateDensity(densityCells, binaryImage=True)

plt.imshow(img, cmap='gray')
plt.imshow(a.processed_png)

fig, ax = dc.create_interactive_heatmap(a.masked, cells, binaryDensities, kernelSize, threshold, 20)

plt.show()