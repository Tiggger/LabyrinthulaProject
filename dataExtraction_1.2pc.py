import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-28_singleCellResolving_20xwithExtender_BF_featureImages_mixofExposuretimes_exports/2025-07-28_singleCellResolving+20xwithExtender_BF_featureImages_mixofExposuretimes_MMStack_Default.ome.jpg'
#xsplit = 6
#ysplit = 6
#kernelSize = 4
#threshold = 128

#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-28_singleCellResolving_20xwithExtender_BF_featureImages_mixofExposuretimes_exports/2025-07-28_singleCellResolving+20xwithExtender_BF_featureImages_mixofExposuretimes_MMStack_Default.ome2.jpg'
#xsplit = 6
#ysplit = 6
#kernelSize=6
#threshold=110

#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-28_singleCellResolving_20xwithExtender_BF_featureImages_mixofExposuretimes_exports/2025-07-28_singleCellResolving+20xwithExtender_BF_featureImages_mixofExposuretimes_MMStack_Default.ome4.jpg'
#xsplit = 6
#ysplit = 6
#kernelSize=8
#threshold=90

#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-28_singleCellResolving_20xwithExtender_BF_featureImages_mixofExposuretimes_exports/2025-07-28_singleCellResolving+20xwithExtender_BF_featureImages_mixofExposuretimes_MMStack_Default.ome5.jpg'
#xsplit = 6
#ysplit = 6
#kernelSize=4
#threshold=150

#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-28_singleCellResolving_20xwithExtender_BF_featureImages_mixofExposuretimes_exports/2025-07-28_singleCellResolving+20xwithExtender_BF_featureImages_mixofExposuretimes_MMStack_Default.ome6.jpg'
#xsplit = 6
#ysplit = 6
#kernelSize=4 #6 works also
#threshold=135

#tile image - ph1, doesn't look brilliant due to difference in cell colour between neigbouring cells
#image_dir='/Volumes/SANDISK/images/nikon3/28.07.25/1.2%/2025-07-25_20xwithExtender_Ph1_singleCell_tileImages_25pcOverlap_1_processed/2025-07-25_20xwithExtender_Ph1_singleCell_tileImages_25pcOverlap_1_timepoint_0.jpg'
#xsplit = 12
#ysplit = 12
#kernelSize=4
#threshold=150

#200 works suprisingly well
image_dir='/Users/johnwhitfield/Desktop/testingLevelChanges.jpg'
xsplit=7
ysplit=7
kernelSize=8
threshold=200


img = np.array(Image.open(image_dir))

cells = dc.splitIntoCells(img, xsplit, ysplit)



a=mc.ImageAnalysis(image_dir, None, 4, kernelSize, threshold)

densityCells = dc.splitIntoCells(a.binary_Image, xsplit, ysplit)

binaryDensities=dc.calculateDensity(densityCells, binaryImage=True)

#plt.imshow(img, cmap='gray')
plt.imshow(a.processed_png)

fig, ax = dc.create_interactive_heatmap(a.masked, cells, binaryDensities, kernelSize, threshold, 20)

plt.show()