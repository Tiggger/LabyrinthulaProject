import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-28_singleCellResolving_20xwithExtender_BF_featureImages_mixofExposuretimes_exports/2025-07-28_singleCellResolving+20xwithExtender_BF_featureImages_mixofExposuretimes_MMStack_Default.ome1.jpg'
#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-28_singleCellResolving_20xwithExtender_BF_featureImages_mixofExposuretimes_exports/2025-07-28_singleCellResolving+20xwithExtender_BF_featureImages_mixofExposuretimes_MMStack_Default.ome.jpg'

image_dir='/Users/johnwhitfield/Desktop/proper/t:19:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

img = np.array(Image.open(image_dir))
xsplit = 10
ysplit = 10
cells = dc.splitIntoCells(img, xsplit, ysplit)

kernelSize=10
threshold=110 #120

#show processed Images - gives good idea of how appropriate image processing looks
a=mc.ImageAnalysis(image_dir, None, 4, kernelSize, threshold)
#shows images
#a.getOrientation('filler')

#calculating density from binary image instead of segmented image
densityCells = dc.splitIntoCells(a.binary_Image, xsplit, ysplit)
#maskedCells = dc.splitIntoCells(a.masked, xsplit, ysplit)

binaryDensities=dc.calculateDensity(densityCells, binaryImage=True)

#fig, ax = dc.create_density_heatmap(a.img, densityCells, binaryDensities)
fig, ax = dc.create_interactive_heatmap(a.masked, cells, binaryDensities, kernelSize, threshold, 20)

#qtensor
#qtensorsInfo=dc.calculateQTensor(cells, kernelSize, threshold)
#print(qtensorsInfo, 'info')

#fig, ax = dc.create_nematicOrderingTensor_heatmap(img, cells, qtensorsInfo, a.masked)

plt.show()