import numpy as np
import sys
import importlib

import masterClass as mc
import imageProcessingTool as dc

import matplotlib.pyplot as plt
from PIL import Image

#image_dir='/Users/johnwhitfield/Desktop/proper/t:19:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

image_dir='/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg'

xsplit=5
ysplit=5
kernelSize=4
threshold=150


#creating synthetic data
#L, d, N = 200, 10, 10
#image_domain, angles_domain = dc.generate_nematic(L, d, N, mode='domain', domain_size=5)
#img=image_domain

#temp_path='/tmp/randomImage.tif'
#Image.fromarray(img).save(temp_path)
#a=mc.ImageAnalysis(temp_path, None, 4, kernelSize, threshold)


#print('loading image')
img = np.array(Image.open(image_dir))

print('splitting into normal cells')
cells = dc.splitIntoCells(img, xsplit, ysplit)



a=mc.ImageAnalysis(image_dir, None, 4, kernelSize, threshold)

plt.imshow(img, cmap='gray')

#qtensor ordering
print('calculating q ordering')
qtensorsInfo=dc.calculateQTensor(cells, kernelSize, threshold)

print('making q map')
#fig, ax = dc.create_nematicOrderingTensor_heatmap(img, cells, qtensorsInfo) a.masked
dc.create_nematicOrderingTensor_heatmap_interactive(a.masked, cells, qtensorsInfo, kernelSize, threshold, 20, coarsening=1000)


plt.show()