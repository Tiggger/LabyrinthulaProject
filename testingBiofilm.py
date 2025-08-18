import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#1
#image_dir='/Users/johnwhitfield/Desktop/fakeBiofilms/fakeBiofilm.tif'

#2
#image_dir='/Users/johnwhitfield/Desktop/fakeBiofilms/fakeBiofilm1.tif'

#3
#image_dir='/Users/johnwhitfield/Desktop/fakeBiofilms/bigFakeBiofilm.tif'

#paper data
image_dir='/Users/johnwhitfield/Desktop/Screenshot 2025-08-14 at 15.15.55.png'

xsplit=5
ysplit=5
pixelSize=0.030344
kernelSize=13

#load in the image
img = np.array(Image.open(image_dir))

#split the image into cells
cells = dc.splitIntoCells(img, xsplit, ysplit)

#create a masterClass object
a=mc.ImageAnalysis(image_dir, None, 4, kernelSize)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(a.img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original', fontsize=20)

ax[1].imshow(a.skeletonImage, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Skeleton', fontsize=20)

#calculate ordering from q tensor
qtensorsInfo=dc.calculateQTensor(cells, kernelSize)

#create the interactive orderig heatmap
#the image for the heatmap can be changed, the masked image from the masterClass object is most useful, but a.rgb (angle map) can be useful to see too
dc.create_nematicOrderingTensor_heatmap_interactive(a.masked, cells, qtensorsInfo, kernelSize, pixelSize, a.colour_wheel, coarsening=100)


plt.show()