# Overview of code

Code developed during an 8 week summer project in the Melaugh Lab at The Univeristy Of Edinburgh in collaboration with the charity Project Seagrass. 
We are interested in the behaviour of the protist Labyrinthula, and this code combines analysis tools created by Joseph Knight with some imaging tools that I have created.

# masterClass

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

Combines image stitching tools, with ordering tools from masterClass. The stitched image can be split into user defined sections (number of cells in x and y), and further analysis is user interactive, allowing for flexibility in what you want to analyse. 
You can create density heatmaps, as well as nematic ordering heatmaps. For the nematic ordering heatmap, the director can also be plotted on the cells. If the user clicks on a cell, it runs the masterClass orientation correlation analysis on that given cell and displays it.
There is also the option to draw a vector between two points, and the code will create the ordering correlation graph for each cell which this vector intersects with.
