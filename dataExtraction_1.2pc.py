import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

"""
LONG EXPOSURES

Look nice
"""
#img 1 
#image_dir='/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

#img 2
#image_dir='/Users/johnwhitfield/Desktop/proper/t:19:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

#img 3 
image_dir='/Users/johnwhitfield/Desktop/proper/t:21:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

"""
SHORT EXPOSURES

Not as good as the long exposures
"""
#img 1
#image_dir='/Users/johnwhitfield/Desktop/proper/shorterExposures/t:1:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_f.jpg'

#img 2 
#image_dir='/Users/johnwhitfield/Desktop/proper/shorterExposures/t:2:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_f.jpg'

#img 3
#image_dir='/Users/johnwhitfield/Desktop/proper/shorterExposures/t:4:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_f.jpg'


"""
BF tile images

Actually look pretty good
"""
#img 1
#image_dir='/Users/johnwhitfield/Desktop/proper/tileImages/2025-07-25_20xwithExtender_singleCell_tileImages_1_MMStack_1-P...5_20xwithExtender_singleCell_tileImages_1_MMStack_1-Pos001_001-1.jpg'

#img 2
#image_dir='/Users/johnwhitfield/Desktop/proper/tileImages/2025-07-25_20xwithExtender_singleCell_tileImages_1_MMStack_1-P...5_20xwithExtender_singleCell_tileImages_1_MMStack_1-Pos002_001-1.jpg'

"""
Video snapshot image

Looks ok
"""
#img 1 
#image_dir='/Users/johnwhitfield/Desktop/proper/videoImages/t:13:60 - 2025-07-25_20xwithExtender_BF_10mintimeLapse_1_MMS.jpg'

"""
Ph1 images

Experimental, as cells are black space, but white space surrounds them
"""
#img 1
#image_dir='/Users/johnwhitfield/Desktop/proper/ph1/2025-07-25_20xwithExtender_Ph1_singleCell_tileImages_25pcOverl...r_Ph1_singleCell_tileImages_25pcOverlap_1_MMStack_1-Pos000_001-1.jpg'

#img 2 
#image_dir='/Users/johnwhitfield/Desktop/proper/ph1/t:15:24 - 2025-07-28_singleCellResolving_20xwithExtender_Ph1.jpg'

#img 3
#image_dir='/Users/johnwhitfield/Desktop/proper/ph1/t:18:24 - 2025-07-28_singleCellResolving_20xwithExtender_Ph1.jpg'

#seem to be pretty good settings
xsplit=7*3
ysplit=7*3
kernelSize=4 #doesn't work well for synthetic data
#kernelSize=10 #good for synthetic data
threshold=128


#creating synthetic data
#L, d, N = 200, 10, 10
#image_domain, angles_domain = dc.generate_nematic(L, d, N, mode='domain', domain_size=5)
#img=image_domain

#temp_path='/tmp/randomImage.tif'
#Image.fromarray(img).save(temp_path)
#a=mc.ImageAnalysis(temp_path, None, 4, kernelSize, threshold)


#print('loading image')
img = np.array(Image.open(image_dir))

#skel = np.array(Image.open(image_dir1))

print('splitting into normal cells')
cells = dc.splitIntoCells(img, xsplit, ysplit)



a=mc.ImageAnalysis(image_dir, None, 4, kernelSize, threshold)

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
qtensorsInfo=dc.calculateQTensor(cells, kernelSize, threshold)

print('making q map')
#fig, ax = dc.create_nematicOrderingTensor_heatmap(img, cells, qtensorsInfo) a.masked (image with angle map), a.rgb (angle map)
dc.create_nematicOrderingTensor_heatmap_interactive(a.masked, cells, qtensorsInfo, kernelSize, threshold, 20, a.colour_wheel ,coarsening=1000)


plt.show()