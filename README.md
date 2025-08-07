# Overview of code

Code developed during an 8 week summer project in the Melaugh Lab at The Univeristy Of Edinburgh in collaboration with the charity Project Seagrass. 
We are interested in the behaviour of the protist Labyrinthula, and this code combines analysis tools created by Joseph Knight with some imaging tools that I have created.

# **masterClass**

A streamlined OOP instance of an ordering correlation analysis tool created by Joseph Knight. Allows for easy application of the functions to data.

  ## __init__

  Computes mostly all of the images and information that is needed for the rest of the functions. These include: loading the image, skeletonising it, and getting the angles of objects within the iamge. This has all been placed here, such that these attributes can be accessed at any time simply by initialising an object from this class definition. This is useful when working with imageProcessingTool.

  ## processSkeleton

  Outputs all of the different images in order of computation. Will output the highlighted endpoints image, which is nor particularly useful for the dense Labyrinthula structures I have been using the code for. Useful for visualising the pipeline and seeing how effective your set kernel size is etc.

  ## getOrientationImages

  Outputs all of the different images which I use for the angle calculation, in order of computation. I.e. the initial image is 'raw', the final image is what it looks like once all the necessary processing has been applied.

  ## calculateOrientationCorrelation

  Looks at the image, compares pixels at given distances and calculates their orientation by looking at the angle that the pipeline has assigned to each pixel and taking a dot product. It does this for all pixels with the same distance between them, and saves all of this information for every comparison it makes. 

  ## binIt

  Bins the distances that were saved in the distance list from calculateOrientationCorrelation, and calculates the average correlation, and the standard deviation for each bin. 

  ## calculateCorrelationAvgNematic

  Calculates the nematic ordering from the bin averages list, ensuring the value lies between 0 and 1.

  ## plotOrientationCorrelation

  Plots the binned correlation averages once data has been passed in. Ensures that the x axis is in units of microns, as opposed to pixels. This gives us a graph of the 'correlation function'.

  ## produceCorrelationGraph

  Calculated all of the necessary things in order for the correlation graph to be plotted correctly. It is made up of a lot of functions that have been written above.

  ## produceBinCentreGraphs

  Produces a plot of the binned distances, and the counts within each. Good for visualising the spread of the data that is being put into the correlation function graph.

# imageProcessingTool

Can pass an image into this code, in order to create density heatmaps (if the images have been segmented in Weka), or a nematic ordering heatmap, which is calculated from the Q Tensor. The code will produce the heatmap, where it will calculate one of the aforementioned features for every image cell which you can set. For example, if you decide you want to split the image into 36 grids, you could split the x and y axis into 6. For each of the 36 grids on the image, the code will calculate the features. If you wish, the code also has the ability to calculate the correlation function for each of the grids. Simply click on the grid with you cursor, and after a few seconds the orientation correlation function will be plotted (calculated from masterClass), along with the image of the cell you hvae clicked on.

  ## splitIntoCells

  Splits the image into smaller images (image cells), such that you can control the 'coarseness' of which you wish to calcualte the nematic ordering and density for. 

  ## calculateDensity

  Calculates the density of each image cell. Works very well if the image you have passed in to the code has been segmented in Weka. This function also has the ability to calculate density from the binary image which is calculated in masterClass. This is a very crude calculation of density, and it not recommended to be used at this current stage.

  ## drawCellBoundariesNumpy

  Once the iamge has been split into the user defined number of image cells, this function will draw the boundaries on the image, such that when the image is displayed, you can see where each image cell lays. 

  ## create_density_heatmap

  Creates a static (non-interactable) density heatmap. Will output the image you have passed in, with the grid lines overlayed to show where the image cells are. Each of these image cells will have a colour overlay with an alpha value (such that is it see through) corresponding to how dense that particular image cell is. A colourbar will be outputted to the right. 

  ## create_interactive_heatmap

  Same output as create_density_heatmap, except when you click on one of the image cells, it will then calculate the ordering correlation function using code from masterClass. The graph will be outputted in a new window, along with the image which was in the image cell you clicked on.

  ## create_interactive_vector_analysis

  Same output as create_density_heatmap, except line vectors can now be drawn for analysis. Once the density heatmap has been outputted, you can click on the first cell you are interested in, and then a second click for the final cell you are interested in. The code will then drawa a line vector between the two click locations, and calculate which cells this line intersects. For each intersected cell, the ordering correlation function will be calculated. All of these graphs will then be outputted. These are not currenlty inputted in order of increasing distance from the inital click, but this feature will be implemented. However, the distance from the initial click is in the title of the plot.

  ## calculateQTensor

  calculates the Q Tensor from the angles which are calculated when running the masterClass code. Once calcualted, it extracts the ordering paramter (S) and the director.

  ## create_nematicOrderingTensor_heatmap_interactive

  Creates a heatmap of nematic ordering. Very similar to the density heatmap, but the differences will be explained now. For each image cell, the nematic ordering is calculated. Instead of a colour overlay to represent S (0 = no ordering, 1 = perfectly ordered), the director (which points in direction of ordering) is plotted at the centre of the image cell, and is scaled to S. Therefore if a small arrow is plotted it means there is little ordering, and vice versa. Therefore, there is no colour bar with this plot. If you click on an image cell, it will calculate the ordering correlation function (from masterClass code) of that particular area. Graph along with the image in that image cell will be outputted in a new window. 

  ## generateNematic

  Has the ability to produce synthetic data. Can produce rods in random directions, in direcctions which change smoothly and slowly with direction, and also domains where the rods are perfectly aligned. This is a very useful function for producing pseudoimages to pass into the pipeline and check how things are working. 
  




  
