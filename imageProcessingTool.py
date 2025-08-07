#necessary imports to get things working
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
import masterClass as mc
import time
import os
import tempfile
import math
import io 
from skimage.transform import resize
import decimal

#Joe code import
import Functions.Orientation as fn

#imports for synthetic data generation
import cv2
from scipy.ndimage import gaussian_filter

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image as PILImage

import sys
#print('\n', sys.path, 'path')

#function to split image into cells
def splitIntoCells(img, xsplit, ysplit):
    
    #to handle if the image has colour data, c not currently used, but could be employed later on
    if len(img.shape)==2:
        rows, cols = img.shape
    elif len(img.shape)==3:
        rows, cols, c = img.shape

    #include colour data but not necessarily going to use
    #rows, cols, c = img.shape

    #calculate the height of a cell
    cellHeight = rows // ysplit
    #calculate the width of a cell
    cellWidth = cols // xsplit

    #list to store cell pixel values
    cells = []

    #looping through the number of y cells
    for y in range(ysplit):
        #keeps cells in a particular row
        rowCells = []

        #from geometry to calculate the start y coord and end y coord of a cell
        yStart = y * cellHeight
        yEnd = (y + 1) * cellHeight if y != ysplit - 1 else rows

        #looping through the number of x cells
        for x in range(xsplit):

            #geometry to calculate the start x coord and end x coord
            xStart = x * cellWidth
            xEnd = (x + 1) * cellWidth if x != xsplit - 1 else cols
            
            #splicing image to create cell
            cell = img[yStart:yEnd, xStart:xEnd]
            #saving this cell to the row cells list
            rowCells.append(cell)
        
        #putting the row of cells in the master cells list
        cells.append(rowCells) 

    return cells

#function to calculate density of a cell in the image
def calculateDensity(cells, binaryImage=False):

    if binaryImage==True:
        off=True #swapped, I think this is correct to be this way
        on=False
    else:
        off=76
        on=188

    #print(cells, 'cells')

    #matrix to store values of density
    densities = []

    #going through rows in the master cells list
    for row in cells:

        #to store the densities of cells in the same row
        row_densities = []
        for cell in row:
            #count the number of each type of pixel. 76 and 188 are pixel values that come out of stitched segmented grey scale image
            count0 = np.sum(cell == off)
            count1 = np.sum(cell == on)
            total = count0 + count1
            
            #Calculate the density of a cell considering fraction of pixels that are cell
            density = count0 / total if total > 0 else 0.0
            #append to row densities list
            row_densities.append(density)
            
            #for debugging
            #print(f"Cell - 76: {count0}, 188: {count1}, Density: {density:.2f}")
        
        #append the density of a row to the master densities list
        densities.append(row_densities)

    return densities

#splits cell into boundaries by drawing lines on top
def drawCellBoundariesNumpy(image, xsplit, ysplit, lineValue=255, thickness=1):
    #create copy of image
    img = image.copy()
    #get shape of image
    height, width = img.shape
    
    #Calculate x coordinates of vertical lines, using geometry
    x_coords = [i * (width // xsplit) for i in range(1, xsplit)]
    #Calculate y coordinates of horizontal lines
    y_coords = [i * (height // ysplit) for i in range(1, ysplit)]
    
    # Draw vertical lines
    for x in x_coords:
        start = max(0, x - thickness//2)
        end = min(width, x + thickness//2 + thickness%2)
        img[:, start:end] = lineValue
    
    # Draw horizontal lines
    for y in y_coords:
        start = max(0, y - thickness//2)
        end = min(height, y + thickness//2 + thickness%2)
        img[start:end, :] = lineValue

    return img

#function to create static density heatmap
def create_density_heatmap(image, cells, densities, cmap='viridis', alpha=0.5):

    #Create figure with two subplots (one for image, one for colourbar)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    #Display the original image in grey scale
    ax.imshow(image, cmap='gray')
    
    #Get cell dimensions from first cell
    cell_height, cell_width = cells[0][0].shape
    
    #Create normalization for the colormap
    norm = Normalize(vmin=0, vmax=1)  # Assuming densities are 0-1
    
    #Create a scalar mappable for the colorbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    #Add coloured rectangles for each cell
    for y in range(len(cells)):
        for x in range(len(cells[0])):

            # Get coordinates for the cell
            x_start = x * cell_width
            y_start = y * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height
            
            # Create a coloured rectangle
            rect = patches.Rectangle(
                (x_start, y_start), cell_width, cell_height,
                linewidth=0,
                facecolor=sm.to_rgba(densities[y][x]),
                alpha=alpha
            )
            #add the colour to the plot
            ax.add_patch(rect)
    
    #Add colourbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density')
    
    #Add grid lines (optional)
    for y in range(len(cells)+1):
        ax.axhline(y * cell_height, color='white', linestyle=':', linewidth=0.5)
    for x in range(len(cells[0])+1):
        ax.axvline(x * cell_width, color='white', linestyle=':', linewidth=0.5)
    
    #plot making
    ax.set_title(f"Density Heatmap ({len(cells[0])}x{len(cells)} grid)")
    ax.axis('off')
    
    #for plotting after calling function
    return fig, ax



#plots a density heatmap, but when you click on a cell, will calculate ordering correlation function using Joe's code
def create_interactive_heatmap(image, cells, densities, kernelSize, threshold, magnification, cmap='viridis', alpha=0.5):
    #create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    #Display the stitched, segmented image in grey scale
    ax.imshow(image, cmap='gray')
    
    #Get cell dimensions
    #to handle if image has colour information, c is not currently used but could be employed later
    if len(cells[0][0].shape)==2:
        cell_height, cell_width = cells[0][0].shape
    elif len(cells[0][0].shape)==3:
        cell_height, cell_width, c = cells[0][0].shape
    
    #Create normalisation and colourmap
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    #Add coloured rectangles for each cell, representing density
    for y in range(len(cells)):
        for x in range(len(cells[0])):
            x_start = x * cell_width
            y_start = y * cell_height
            
            #create colour patch
            rect = patches.Rectangle(
                (x_start, y_start), cell_width, cell_height,
                linewidth=0, facecolor=sm.to_rgba(densities[y][x]), alpha=alpha
            )
            #overlay colour patch
            ax.add_patch(rect)
    
    #Add colourbar
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Density')
    
    #Add grid lines
    for y in range(len(cells)+1):
        ax.axhline(y * cell_height, color='white', linestyle=':', linewidth=0.5)
    for x in range(len(cells[0])+1):
        ax.axvline(x * cell_width, color='white', linestyle=':', linewidth=0.5)
    
    #plotting attributes
    ax.set_title(f"Interactive Density Heatmap - Click on any cell")
    ax.axis('off')
    
    #Click event handler, telling computer to do when you click
    def onclick(event):
        #print to let me know click has been registered
        print('click worked')

        #if click out of bounds
        if event.inaxes != ax:
            return
            
        #Get click coordinates
        x_click, y_click = int(event.xdata), int(event.ydata)
        
        # Determine which cell was clicked
        cell_x = x_click // cell_width
        cell_y = y_click // cell_height
        
        #Safety check, if outside of bounds
        if cell_y >= len(cells) or cell_x >= len(cells[0]):
            return
            
        #Get the clicked cell from the cell list
        clicked_cell = cells[cell_y][cell_x]
        density=densities[cell_y][cell_x]
        
        #Create a temporary image file for the cell, need this due to how masterClass has been written, needs image path
        temp_path = "/tmp/clicked_cell.tif"
        Image.fromarray(clicked_cell).save(temp_path)
        
        #Analyse the cell, putting into masterClass
        cell_analysis = mc.ImageAnalysis(
            imagePath=temp_path,
            skeletonImagePath=None,
            radius=4,
            sl=kernelSize, 
            threshold=threshold
        )
        
        # Create a new figure for the correlation graph
        plt.figure(figsize=(10, 4))

        
        # Show the cell image
        plt.subplot(1, 2, 1)
        #shows image in cell we have clicked on 
        plt.imshow(clicked_cell, cmap='gray')
        plt.title(f"Cell ({cell_x}, {cell_y}), Density: {round(density, 3)}")
        plt.axis('off')
        
        # Show the correlation graph
        plt.subplot(1, 2, 2)
        #shows the calculated correlation graph for the given cell
        cell_analysis.produceCorrelationGraph(
            #changed from 10000 to 1000
            coarsening=1000,
            title=f"Orientation Correlation - Cell ({cell_x}, {cell_y})",
            magnification=magnification,
            bin_size=2
        )
        
        #show off the plots
        plt.tight_layout()
        plt.show()
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    #show everything
    plt.tight_layout()
    plt.show()

#Extra updated version, which allows us to draw lines to calculate the ordering for several cells and display
def create_interactive_vector_analysis(image, cells, densities, cmap='viridis', alpha=0.5):

    #create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Display the original image in greyscale
    ax.imshow(image, cmap='gray')
    
    #Work out cell dimensions
    cell_height, cell_width = cells[0][0].shape
    img_height, img_width = image.shape
    
    #Create normalisation and colourmap
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    #Add coloured rectangle overlay for each cell
    #going over cells
    for y in range(len(cells)):
        for x in range(len(cells[0])):
            x_start = x * cell_width
            y_start = y * cell_height
            
            #create rectangle
            rect = patches.Rectangle(
                (x_start, y_start), cell_width, cell_height,
                linewidth=0, facecolor=sm.to_rgba(densities[y][x]), alpha=alpha
            )
            #add patch to plot
            ax.add_patch(rect)
    
    #Add colourbar
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label='Density')
    
    # Add grid lines
    for y in range(len(cells)+1):
        ax.axhline(y * cell_height, color='white', linestyle=':', linewidth=0.5)
    for x in range(len(cells[0])+1):
        ax.axvline(x * cell_width, color='white', linestyle=':', linewidth=0.5)
    
    #telling how to use
    ax.set_title("Click to set start point, then end point (right-click to cancel)")
    ax.axis('off')
    
    # Variables to store click points
    start_point = None
    end_point = None
    line = None
    
    def onclick(event):
        #define variables nonlocal to use within this subfunction
        nonlocal start_point, end_point, line
        
        #if out of bounds
        if event.inaxes != ax:
            return
            
        #Right click cancels current selection
        if event.button == 3:
            if line:
                line.remove()
                line = None
            start_point = None
            end_point = None
            #remind them how to use
            ax.set_title("Click to set start point, then end point (right-click to cancel)")
            fig.canvas.draw()
            return
            
        #Get click coordinates
        x_click, y_click = int(event.xdata), int(event.ydata)
        
        #First click sets start point
        if start_point is None:
            start_point = (x_click, y_click)
            #update title to instruct user on next move
            ax.set_title("Now click end point (right-click to cancel)")
            fig.canvas.draw()
            return
            
        #Second click sets end point
        end_point = (x_click, y_click)
        
        # Draw the vector line
        #if there is already line, remove it
        if line:
            line.remove()
        
        #draw the new line
        line, = ax.plot([start_point[0], end_point[0]], 
                    [start_point[1], end_point[1]], 
                    'r-', linewidth=2)
        fig.canvas.draw()
        
        #Find and analyse cells along the vector path
        analyse_cells_along_vector(start_point, end_point, cells, cell_width, cell_height)
        
        #Reset for next selection
        start_point = None
        end_point = None
        ax.set_title("Click to set start point, then end point (right-click to cancel)")
    


    def analyse_cells_along_vector(start, end, cells, cell_w, cell_h):
        #unpacking start and end coordinates
        x0, y0 = start
        x1, y1 = end
        
        #Get ordered cells along path, check for which cells we are interested in
        cells_along_path = get_cells_along_line(x0, y0, x1, y1, cell_w, cell_h,
                                            len(cells[0]), len(cells))
        
        #line for debugging
        #print(f"\nAnalyzing {len(cells_along_path)} cells along path:")
        
        #Store analysis results for later plotting
        analysis_results = []
        
        #going through cells we are interested in
        for i, (cell_x, cell_y) in enumerate(cells_along_path):
            cell = cells[cell_y][cell_x]
            
            #Create temporary file, do this due to how masterClass is written
            temp_path = f"/tmp/cell_{cell_x}_{cell_y}.tif"
            Image.fromarray(cell).save(temp_path)
            
            
            #Initialise analysis object
            analysis = mc.ImageAnalysis(
                imagePath=temp_path,
                skeletonImagePath=None,
                radius=4,
                sl=4
            )
            
            #append result which comes from produceCorrelationGraph, new variable passed into which is plotting
            #previously the function would always plot, but want to be able to control when 
            analysis_results.append(analysis.produceCorrelationGraph(
                coarsening=10000,
                title=f"Cell {i+1} at ({cell_x}, {cell_y})",
                bin_size=2, 
                plotting=False
            ))

        #line for debugging
        #print(analysis_results, 'analysisresults')
        
        #number of cells to plot
        num_cells = len(analysis_results)
    
        #Create figure with one row of subplots
        fig, axs = plt.subplots(1, num_cells, figsize=(4*num_cells, 4))
        if num_cells == 1:
            axs = [axs]  #Ensure axs is always a list
    
        #plotting data
        for i, ((bin_centers, corr_data), (cell_x, cell_y)) in enumerate(zip(analysis_results, cells_along_path)):
            #Calculate distance from start point, want to update it such that we plot data in order of increasing distance
            distance = math.hypot(cell_x*cell_w - start_point[0], cell_y*cell_h - start_point[1])
            
            #Plot correlation data
            axs[i].plot(bin_centers, corr_data, '-o', markersize=3)
            axs[i].set_xlabel('Distance from center of cell')
            axs[i].set_ylabel('Correlation')
            axs[i].set_title(f"Cell {i+1}\nPos: ({cell_x},{cell_y})\nDist: {distance:.1f}px")
            axs[i].grid(True)
            
            
        #show the data
        plt.tight_layout()
        plt.show()
        
    
    #getting cells along line, using well known algorithm. Pass in start and end points, cell width and height as well as grid dimensions
    def get_cells_along_line(x0, y0, x1, y1, cell_w, cell_h, grid_w, grid_h):
        """Bresenham's line algorithm modified to return unique cells along path"""
        cells = set()
        
        #finding change in x and y to work with
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0

        #not sure
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        
        #if increase in x bigger than y
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                #Add current cell
                cell_x = x // cell_w
                cell_y = y // cell_h
                if 0 <= cell_x < grid_w and 0 <= cell_y < grid_h:
                    cells.add((cell_x, cell_y))
                
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        #if the opposite is true
        else:
            err = dy / 2.0
            while y != y1:
                #Add current cell
                cell_x = x // cell_w
                cell_y = y // cell_h
                if 0 <= cell_x < grid_w and 0 <= cell_y < grid_h:
                    cells.add((cell_x, cell_y))
                
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        
        #Add final cell
        cell_x = x1 // cell_w
        cell_y = y1 // cell_h
        if 0 <= cell_x < grid_w and 0 <= cell_y < grid_h:
            cells.add((cell_x, cell_y))
        
        #Convert to ordered list from start to end
        ordered_cells = []
        if cells:
            #Calculate distances from start point and sort
            cells_list = list(cells)
            distances = [math.hypot((x//cell_w)*cell_w - x0, (y//cell_h)*cell_h - y0) for x, y in cells_list]
            ordered_cells = [cell for _, cell in sorted(zip(distances, cells_list))]
        
        return ordered_cells
    
    #Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    #show it all
    plt.tight_layout()
    plt.show()




#nematic ordering section

#this function is computationally quite heavy, and takes a long while to run
def calculateOrdering(cells, kernelSize, threshold):
    #place to store results
    ordering=[]

    #going through cells
    for y in range(len(cells)):
        row_ordering=[]
        for x in range(len(cells[y])):
            #get cell
            cell=cells[x][y]


            #Create a temporary file path, needed for masterClass
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"cell_{y}_{x}.tif")
            
            #Save cell as image
            Image.fromarray(cell).save(temp_path)

            #create object to get angles
            cell_analysis=mc.ImageAnalysis(temp_path, None, 4, kernelSize, threshold)

            #line for debugging
            #print(cell_analysis.phi_new, 'phi_new')

            #eigenvalue of ordering tensor - I think that this is the right angle list
            orderingAverage=(np.average((3*np.cos(cell_analysis.phi)**2-1)/(2)))
            #print(orderingAverage, 'orderingAverage')

            #building row of ordering values within the grid
            row_ordering.append(float(orderingAverage))

        #put row ordering into matrix that holds ordering of all positions in grid
        ordering.append(row_ordering)

    return ordering

#analogous to density heatmap, just with ordering plotting instead
def create_nematicOrdering_heatmap(image, cells, orderings, cmap='viridis', alpha=0.5):
   
    #Create figure with two subplots (one for image, one for colourbar)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    #Display the original image
    ax.imshow(image, cmap='gray')
    
    #Get cell dimensions from first cell
    cell_height, cell_width = cells[0][0].shape
    
    #Create normalisation for the colourmap
    norm = Normalize(vmin=0, vmax=1)  #Assuming densities are 0-1
    
    #Create a scalar mappable for the colourbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    #Add coloured rectangles for each cell
    for y in range(len(cells)):
        for x in range(len(cells[0])):
            #Get coordinates for this cell
            x_start = x * cell_width
            y_start = y * cell_height
            x_end = x_start + cell_width
            y_end = y_start + cell_height
            
            #Create a coloured rectangle patch to be overlayed with some alpha
            rect = patches.Rectangle(
                (x_start, y_start), cell_width, cell_height,
                linewidth=0,
                facecolor=sm.to_rgba(orderings[y][x]),
                alpha=alpha
            )
            #add patch to plot
            ax.add_patch(rect)
    
    # Add colourbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Nematic Ordering')
    
    #Add grid lines
    for y in range(len(cells)+1):
        ax.axhline(y * cell_height, color='white', linestyle=':', linewidth=0.5)
    for x in range(len(cells[0])+1):
        ax.axvline(x * cell_width, color='white', linestyle=':', linewidth=0.5)
    
    #plot attributes
    ax.set_title(f"Nematic Ordering Heatmap ({len(cells[0])}x{len(cells)} grid)")
    ax.axis('off')
    
    #return to plot as i wish
    return fig, ax


#calculating ordering from qtensor
def calculateQTensor(cells, kernelSize, threshold, batch_size=1000):
    info = []
    
    for y in range(len(cells)):
        rowInfo=[]
        for x in range(len(cells[y])):
            cell = cells[y][x]
            
            # Save cell as image (keeping your existing code)
            temp_dir = tempfile.gettempdir()
            temp_path = os.path.join(temp_dir, f"cell_{y}_{x}.tif")
            Image.fromarray(cell).save(temp_path)
            
            # Create analysis object
            cell_analysis = mc.ImageAnalysis(temp_path, None, 4, kernelSize, threshold)
            #getting relevant information - originally .phi, but should it be .norm_phi? Been using phi_new
            angles = cell_analysis.phi_new

            #angle debugging

            #print("Min angle (rad):", np.min(angles))
            #print("Max angle (rad):", np.max(angles))

            #to check angles are in typical range, appear to be ok
            #print(angles/np.pi)
            
            # Initialize accumulators for outer product components
            sum_xx = 0.0
            sum_xy = 0.0
            sum_yy = 0.0
            total = len(angles)
            
            # Process angles in batches to avoid memory overload
            for i in range(0, total, batch_size):
                batch_angles = angles[i:i+batch_size]
                
                # Calculate cos and sin for the batch
                cos_theta = np.cos(batch_angles)
                sin_theta = np.sin(batch_angles)

                #debugging
                #norms = np.sqrt(cos_theta**2 + sin_theta**2)  # Should be ~1.0
                #print("Max norm:", np.max(norms), "Min norm:", np.min(norms))
                
                # Accumulate components directly
                sum_xx += np.sum(cos_theta * cos_theta)
                sum_xy += np.sum(cos_theta * sin_theta)
                sum_yy += np.sum(sin_theta * sin_theta)
            
            # Compute averages
            avg_xx = np.mean(cos_theta**2)    # ⟨cos²θ⟩
            avg_xy = np.mean(cos_theta * sin_theta)  # ⟨cosθsinθ⟩
            avg_yy = np.mean(sin_theta**2)    # ⟨sin²θ⟩
            
            
            # Construct Q tensor directly
            q_tensor = np.array([
                [2 * avg_xx - 1, 2 * avg_xy],
                [2 * avg_xy, 2 * avg_yy - 1]
            ])
            
            #rowInfo.append(q_tensor)
            eigenvalues, eigenvectors = np.linalg.eig(q_tensor)
            S = np.max(eigenvalues)
            director = eigenvectors[:, np.argmax(eigenvalues)]

            #ridding of artificial ordering of blank cells, don't expect higher than ~0.8
            if S >= 0.8:
                S=0

            rowInfo.append([S, director])
        
        info.append(rowInfo)
    
    return info

def create_nematicOrderingTensor_heatmap(image, cells, orderingInfo, masked_image=None, arrow_scale=0.3, arrows=True, cmap='viridis', alpha=0.5):
    #Create figure with two subplots (one for image, one for colourbar)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Display either the original or masked image
    if masked_image is not None:
        ax.imshow(masked_image)  # RGB masked image
    else:
        ax.imshow(image, cmap='gray')

    #Get cell dimensions from first cell
    #check for colour information and handle appropriately 
    if len(cells[0][0].shape)==2:
        cell_height, cell_width = cells[0][0].shape
    elif len(cells[0][0].shape)==3:
        cell_height, cell_width, c = cells[0][0].shape

    #calculating the size of the arrows to be plotted
    arrow_length = min(cell_width, cell_height) * arrow_scale
    
    #Create normalisation for the colourmap
    norm = Normalize(vmin=0, vmax=1)  #Assuming densities are 0-1
    
    #Create a scalar mappable for the colourbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    #Add coloured rectangles for each cell
    for y in range(len(cells)):
        for x in range(len(cells[0])):
            # Get coordinates for this cell (center point)
            x_center = x * cell_width + cell_width/2
            y_center = y * cell_height + cell_height/2
            
            # Create a coloured rectangle patch
            rect = patches.Rectangle(
                (x * cell_width, y * cell_height), cell_width, cell_height,
                linewidth=0,
                facecolor=sm.to_rgba(orderingInfo[y][x][0]),
                alpha=alpha
            )
            ax.add_patch(rect)
            
            # Add director arrow if requested
            if arrows:
                S, director = orderingInfo[y][x]

                #handling if negative, for plotting
                #if director[0]<0:
                #    director[0]*=-1

                dx = director[0] * arrow_length  # x-component of arrow
                dy = -director[1] * arrow_length  # y-component (negative because image y-axis is inverted)
                
                ax.arrow(
                    x_center, y_center, 
                    dx, dy,
                    head_width=arrow_length*0.3,
                    head_length=arrow_length*0.4,
                    fc='white', ec='red',
                    linewidth=1.5,
                    length_includes_head=True
                )
    
    # Add colourbar
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Nematic Ordering')
    
    #Add grid lines
    for y in range(len(cells)+1):
        ax.axhline(y * cell_height, color='white', linestyle=':', linewidth=0.5)
    for x in range(len(cells[0])+1):
        ax.axvline(x * cell_width, color='white', linestyle=':', linewidth=0.5)
    
    #plot attributes
    ax.set_title(f"Nematic Ordering Heatmap ({len(cells[0])}x{len(cells)} grid)")
    ax.axis('off')
    
    #return to plot as i wish
    return fig, ax

#qtensor nematic ordering map, but with interactive calculation 
def create_nematicOrderingTensor_heatmap_interactive(image, cells, orderingInfo, kernelSize, threshold, magnification, colourWheel, coarsening=10000, masked_image=None, arrow_scale=0.3, arrows=True, cmap='viridis', alpha=0.5):
    #Create figure with two subplots (one for image, one for colourbar)
    fig, (ax, axWheel) = plt.subplots(1, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [3, 1]})

    # Display either the original or masked image
    if masked_image is not None:
        ax.imshow(masked_image)  # RGB masked image
    else:
        ax.imshow(image, cmap='gray')

    #Get cell dimensions from first cell
    #check for colour information and handle appropriately 
    if len(cells[0][0].shape)==2:
        cell_height, cell_width = cells[0][0].shape
    elif len(cells[0][0].shape)==3:
        cell_height, cell_width, c = cells[0][0].shape

    #calculating the size of the arrows to be plotted - only used for length of arrow head right now
    arrow_length = min(cell_width, cell_height) * arrow_scale
    
    #Create normalisation for the colourmap
    norm = Normalize(vmin=0, vmax=1)  #Assuming densities are 0-1
    
    #Create a scalar mappable for the colourbar
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    #Add coloured rectangles for each cell
    for y in range(len(cells)):
        for x in range(len(cells[0])):
            # Get coordinates for this cell (center point)
            x_center = x * cell_width + cell_width/2
            y_center = y * cell_height + cell_height/2
            
            
            
            # Add director arrow if requested
            if arrows:
                S, director = orderingInfo[y][x]

                #handling if negative, for plotting
                #if director[0]<0:
                #    director[0]*=-1

                #replaced arrow length with S, scaled by dimensions of box, divided by 2 s.t. the arrow remains within the dimensions of the box
                dx = director[0] * S * (cell_width/2)  # x-component of arrow
                dy = -director[1] * S * (cell_height/2)  # y-component (negative because image y-axis is inverted)
                
                ax.arrow(
                    x_center, y_center, 
                    dx, dy,
                    head_width=S*0.3*(cell_width/2), #scaled by cell width 
                    head_length=S*0.4*(cell_height/2), #scaled by cell height
                    fc='white', ec='white',
                    linewidth=1.5,
                    length_includes_head=True
                )
    
    # Add colourbar
    #cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    #cbar.set_label('Nematic Ordering')

    axWheel.imshow(colourWheel, extent=[-1, 1, -1, 1])
    axWheel.axis('off')

    # Add angle labels around the wheel
    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    for angle_index in angles:
        x_text = 1.1 * np.cos(angle_index)
        y_text = 1.1 * np.sin(angle_index)
        angle_degrees = np.degrees(angle_index)
        axWheel.text(x_text, y_text, f"{int(angle_degrees)}°", ha='center', va='center')
    
    #Add grid lines
    for y in range(len(cells)+1):
        ax.axhline(y * cell_height, color='white', linestyle=':', linewidth=0.5)
    for x in range(len(cells[0])+1):
        ax.axvline(x * cell_width, color='white', linestyle=':', linewidth=0.5)
    
    #plot attributes
    ax.set_title(f"Nematic Ordering Heatmap ({len(cells[0])}x{len(cells)} grid)")
    ax.axis('off')
    
    #Click event handler, telling computer to do when you click
    def onclick(event):
        #print to let me know click has been registered
        print('click worked')

        #if click out of bounds
        if event.inaxes != ax:
            return
            
        #Get click coordinates
        x_click, y_click = int(event.xdata), int(event.ydata)
        
        # Determine which cell was clicked
        cell_x = x_click // cell_width
        cell_y = y_click // cell_height
        
        #Safety check, if outside of bounds
        if cell_y >= len(cells) or cell_x >= len(cells[0]):
            return
            
        #Get the clicked cell from the cell list
        clicked_cell = cells[cell_y][cell_x]
        ordering=orderingInfo[cell_y][cell_x][0] #could be orderingInfo...[1] instead
        
        #Create a temporary image file for the cell, need this due to how masterClass has been written, needs image path
        temp_path = "/tmp/clicked_cell.tif"
        Image.fromarray(clicked_cell).save(temp_path)
        
        #Analyse the cell, putting into masterClass
        cell_analysis = mc.ImageAnalysis(
            imagePath=temp_path,
            skeletonImagePath=None,
            radius=4,
            sl=kernelSize, 
            threshold=threshold
        )
        
        # Create a new figure for the correlation graph
        plt.figure(figsize=(10, 4))

        
        # Show the cell image
        plt.subplot(1, 2, 1)
        #shows image in cell we have clicked on 
        plt.imshow(clicked_cell, cmap='gray')
        plt.title(f"Cell ({cell_x}, {cell_y}), Ordering (S): {round(ordering, 3)}")
        plt.axis('off')
        
        # Show the correlation graph
        plt.subplot(1, 2, 2)
        #shows the calculated correlation graph for the given cell
        cell_analysis.produceCorrelationGraph(
            #changed from 10000 to 1000
            coarsening=coarsening,
            title=f"Orientation Correlation - Cell ({cell_x}, {cell_y})",
            magnification=magnification,
            bin_size=2
        )
        
        #show off the plots
        plt.tight_layout()
        plt.show()

    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    #show everything
    plt.tight_layout()
    plt.show()


#synthetic data generator - good for testing.
def generate_nematic(L, d, N, mode='random', domain_size=None, correlation_length=5):
    #generating matrix of zeros, faster than creating image on the fly
    image = np.zeros((L, L), dtype=np.uint8)
    spacing = L // N  # Grid spacing for rods
    center_offset = spacing // 2  # Center rods in grid cells
    
    #instructs how to orient rod on the corresponding position in the image matrix
    orientations = np.zeros((N, N))
    
    #handles when mode is set to random, picks a value between 0 and 4pi (why not 2 pi), and assigns every position in image matrix, 
    #originally 0, 4*np.pi
    if mode == 'random':
        orientations = np.random.uniform(-np.pi, np.pi, (N, N))
    
    #handles when the mode is set to smooth
    elif mode == 'smooth':
        #randomly assign orientations between ±40pi to every position 
        orientations = np.random.uniform(-40*np.pi, 40*np.pi, (N, N))  # Start with random noise
        #blend the change in orientation between each orientation using gaussian blur
        orientations = gaussian_filter(orientations, sigma=correlation_length)  # Apply smoothing filter
    
    #handles when we want domains in the image
    elif mode == 'domain' and domain_size is not None: #ensure that domain size has been given
        for i in range(0, N, domain_size):
            for j in range(0, N, domain_size): #nested loop to access every coordinate on the grid
                angle = np.random.uniform(0, np.pi) #generate angle between 0 and pi (rod is symmetric and non-polar, so no need to generate from 2pi)
                orientations[i:i+domain_size, j:j+domain_size] = angle #sets the orientations from i, up to the size of the domain as the same generated angle
                #unclear to me why the above doesn't overwrite the domain size right now
    
    #generating image
    for i in range(N):
        for j in range(N):
            cx, cy = i * spacing + center_offset, j * spacing + center_offset #calculates center of rod to plot for every coordinate
            angle = orientations[i, j] #accesses correct angle
            dx, dy = int((d / 2) * np.cos(angle)), int((d / 2) * np.sin(angle)) #trig to calculate some change 
            cv2.line(image, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, 1) #image where you want to draw, start coord, end coord, colour, thickness
    
    return image, orientations


#-------------------------------------

#USING THE FUNCTIONS BELOW

#-------------------------------------





# Example usage
"""
#Testing usage of ordering
image_dir = '/Users/johnwhitfield/Desktop/2025-07-14_10x_BF_tile_1_timepoint_0_16bit.tif'
img = np.array(Image.open(image_dir))
xsplit = 100
ysplit = 100
cells = splitIntoCells(img, xsplit, ysplit)

print('going to calculate orderings')

orderings=calculateOrdering(cells)

print('finished that, plotting now')


fig, ax = create_nematicOrdering_heatmap(img, cells, orderings, cmap='plasma', alpha=0.6)

plt.show()

"""

"""

image_dir = '/Users/johnwhitfield/Desktop/2025-07-14_10x_BF_tile_1_timepoint_0_16bit.tif'
img = np.array(Image.open(image_dir))

xsplit = 20
ysplit = 20
cells = splitIntoCells(img, xsplit, ysplit)

#imageWithLines = drawCellBoundariesNumpy(img, xsplit, ysplit, thickness=100)
#plt.imshow(imageWithLines, cmap='Greys_r')
#plt.show()

# Calculate densities for all cells
densities = calculateDensity(cells)

# Print the density matrix
#print("\nDensity Matrix:")
#for row in densities:
#    print([f"{d:.2f}" for d in row])

#matplotlib_overlay(img, cells, densities)

#create_interactive_heatmap(img, cells, densities, cmap='plasma', alpha=0.6)
create_interactive_vector_analysis(img, cells, densities, cmap='plasma', alpha=0.6)

#fig, ax = create_density_heatmap(img, cells, densities, cmap='plasma', alpha=0.6)

#plt.show()

"""