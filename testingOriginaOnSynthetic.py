import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#creating synthetic data
L, d, N = 1000, 40, 40
image_domain, angles_domain = dc.generate_nematic(L, d, N, mode='smooth', domain_size=5)
img=image_domain

xsplit, ysplit = 10,10


temp_path='/tmp/randomImage.tif'
Image.fromarray(img).save(temp_path)
a=mc.ImageAnalysis(temp_path, None, 4, 4)


#print('loading image')
#img = np.array(Image.open(image_dir))

#skel = np.array(Image.open(image_dir1))

print('splitting into normal cells')
cells = dc.splitIntoCells(img, xsplit, ysplit)



#a=mc.ImageAnalysis(image_dir, None, 4, kernelSize, threshold)

#showing original, andskeletonised image that has been done in the pipeline

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4), sharex=True, sharey=True)

ax = axes.ravel()

ax[0].imshow(a.img, cmap=plt.cm.gray)
ax[0].axis('off')
ax[0].set_title('original', fontsize=20)

ax[1].imshow(a.skeletonImage, cmap=plt.cm.gray)
ax[1].axis('off')
ax[1].set_title('skeleton', fontsize=20)

fig.tight_layout()

#qtensor ordering
print('calculating q ordering')
qtensorsInfo=dc.calculateQTensor(cells, 4)



print('making q map')
#fig, ax = dc.create_nematicOrderingTensor_heatmap(img, cells, qtensorsInfo) a.masked (image with angle map), a.rgb (angle map)
dc.create_nematicOrderingTensor_heatmap_interactive(a.masked, cells, qtensorsInfo, 4, 20, a.colour_wheel, coarsening=100)


plt.show()