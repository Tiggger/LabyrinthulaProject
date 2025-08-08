import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#import image - download image from repo and change path as needed
#image_dir='/Users/johnwhitfield/Desktop/LabyrinthulaProject/t:21:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

#this one is much better
image_dir='/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

#setting dimensions of grid
xsplit=5
ysplit=5
#setting the size of the gaussian kernel, must be changed for different magnfications. 4 is good for 20x, would need to be increased for lesser magnifications
kernelSize=4 



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
ax[0].set_title('original', fontsize=20)

ax[1].imshow(a.skeletonImage, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()

#calculate ordering from q tensor

qtensorsInfo=dc.calculateQTensor(cells, kernelSize)

#create the interactive orderig heatmap
#the image for the heatmap can be changed, the masked image from the masterClass object is most useful, but a.rgb (angle map) can be useful to see too
dc.create_nematicOrderingTensor_heatmap_interactive(a.masked, cells, qtensorsInfo, kernelSize, 20, a.colour_wheel, coarsening=1000)


plt.show()