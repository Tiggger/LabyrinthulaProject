# Overview of code

Code developed during an 8 week summer project in the Melaugh Lab at The Univeristy Of Edinburgh in collaboration with the charity Project Seagrass. 
We are interested in the behaviour of the protist Labyrinthula, and this code combines analysis tools created by Joseph Knight with some imaging tools that I have created.

# masterClass

A streamlined OOP instance of an ordering correlation analysis tool created by Joseph Knight. Allows for easy application of the functions to data.

  # __init__

  Testing writing under here.

# imageProcessingTool

Combines image stitching tools, with ordering tools from masterClass. The stitched image can be split into user defined sections (number of cells in x and y), and further analysis is user interactive, allowing for flexibility in what you want to analyse. 
You can create density heatmaps, as well as nematic ordering heatmaps. For the nematic ordering heatmap, the director can also be plotted on the cells. If the user clicks on a cell, it runs the masterClass orientation correlation analysis on that given cell and displays it.
There is also the option to draw a vector between two points, and the code will create the ordering correlation graph for each cell which this vector intersects with.
