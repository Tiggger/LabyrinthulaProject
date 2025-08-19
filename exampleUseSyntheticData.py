import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#analysis parameters
#seem to be pretty good settings
xsplit=5*3
ysplit=5*3
kernelSize=4 

pixelSize=0.5 #arbitrary value


#create synthetic data
L, d, N = 1000, 40, 40
image_domain, angles_domain = dc.generate_nematic(L, d, N, mode='smooth', domain_size=5)
img=image_domain

#create synthetic data temporary image path (needed as image analysis needs path to image to be used)
temp_path='/tmp/randomImage.tif'
Image.fromarray(img).save(temp_path)
a=mc.ImageAnalysis(temp_path, None, 4, kernelSize)

#split the image into cells
cells = dc.splitIntoCells(img, xsplit, ysplit)

#showing the image we pass in and the skeletonised image that has been created in the pipeline
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(a.img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original', fontsize=20)

ax[1].imshow(a.skeletonImage, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Skeleton', fontsize=20)

plt.show()

#calculate ordering from q tensor
#change intensity threshold as most of image for synthetic data is dark 
qtensorsInfo=dc.calculateQTensor(cells, kernelSize*2, intensityThreshold=0.01)
#print(qtensorsInfo, 'qtensorInfo')

#create the interactive ordering heatmap
dc.create_nematicOrderingTensor_heatmap_interactive(a.masked, cells, qtensorsInfo, kernelSize, pixelSize, a.colour_wheel, coarsening=1000)
