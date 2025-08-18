import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#import image - download image from repo and change path as needed
#nikon 3
#1
image_dir='/Users/johnwhitfield/Desktop/t_21_21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

#2
#image_dir='/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

#nikon 1
#20x
#image_dir='/Users/johnwhitfield/Desktop/reprocessing/20x/t:5:12 - 2025-08-12_20x_BF_featureImages_MMStack_Pos0.jpg'

#40x
#image_dir='/Users/johnwhitfield/Desktop/reprocessing/40x/t:28:33 - 2025-08-12_40x_BF_featureImages_MMStack_Pos0.jpg'

#setting dimensions of grid
xsplit=2
ysplit=2
#setting the size of the gaussian kernel, must be changed for different magnfications. 4 is good for 20x, would need to be increased for lesser magnifications
kernelSize=4

#defining pixelSizes
nikon1_20x_pixelSize=0.55
nikon1_40x_pixelSize=0.275

nikon3_4x_pixelSize=1.6169
nikon3_10x_pixelSize=0.651
nikon3_20x_pixelSize=0.3236

#load in the image
img = np.array(Image.open(image_dir))

#split the image into cells
cells = dc.splitIntoCells(img, xsplit, ysplit)

#create a masterClass object
a=mc.ImageAnalysis(image_dir, None, 4, kernelSize)

#showing the image we pass in and the skeletonised image that has been created in the pipeline
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(a.img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('Original', fontsize=20)

ax[1].imshow(a.skeletonImage, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('Skeleton', fontsize=20)

"""
ax[2].imshow(a.rgb)
ax[2].axis('off')
ax[2].set_title('Orientation Field', fontsize=20)



ax[3].imshow(a.phi_new, cmap='hsv')
ax[3].axis('off')
ax[3].set_title('All Angles', fontsize=20)

ax[4].imshow(np.where(a.binary_Image, a.phi, np.nan), cmap='hsv')
ax[4].axis('off')
ax[4].set_title('Masked Angles', fontsize=20)
"""



#calculate ordering from q tensor
qtensorsInfo=dc.calculateQTensor(cells, kernelSize)

#create the interactive orderig heatmap
#the image for the heatmap can be changed, the masked image from the masterClass object is most useful, but a.rgb (angle map) can be useful to see too
dc.create_nematicOrderingTensor_heatmap_interactive(a.masked, cells, qtensorsInfo, kernelSize, nikon1_20x_pixelSize, a.colour_wheel, coarsening=1000)


plt.show()