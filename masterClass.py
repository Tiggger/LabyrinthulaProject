#necessary imports
from PIL import Image # type: ignore
import matplotlib.pyplot as plt # type: ignore
import numpy as np # type: ignore
import importlib
import os
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

import Functions.ProcessSkeleton as skel
import Functions.Orientation as fn
import Functions.Dilation as dl
import Functions.Pipeline as pipe

import sys
#print('\n', sys.path, 'path')

class ImageAnalysis():
    #sorting variables, and formatting image
    #expecting magnification to be entered in as string '20x' etc, but perhaps number is better
    def __init__(self, imagePath, skeletonImagePath, radius, sl, threshold):
        self.imagePath = imagePath
        self.skeletonImagePath = skeletonImagePath
        self.radius = radius #for end points of cells (network)
        self.sl = sl
        self.threshold=threshold
    
        #formatting image, and converting to greyscale
        self.img = Image.open(self.imagePath).convert("L")

        #check if skeleton image has been entered
        if skeletonImagePath!=None:
            self.skeletonImage = Image.open(self.skeletonImagePath).convert("L")
        else:
            self.skeletonImage = None
        
        #creating binary image
        self.binary_Image = np.array(self.img) > self.threshold #128 included for thresholding

        #processes the PNG skeleton - moved from processSkeleton so that can be accessed at all times
        self.processed_png, self.highlighted_png, self.endpoints_png = skel.process_skeleton(self.binary_Image.astype(int), A=10)

        #calculating the angle and ordering parameter - moved from getOrientation to be accessed at all times
        self.phi, self.nop = fn.OrientationFilter(np.array(self.processed_png), self.sl)

        #checking angles - moved from getOrientation so can be accessed at all times
        self.phi_rotated = fn.rotate_and_wrap_angles(self.phi,theta = np.pi/2)
        self.phi_new = fn.update_angles(self.phi,epsilon = 0.1,theta = np.pi/2)

        #normalising corrected angles - moved from getOrientation so can be accessed at all times
        self.norm_phi = fn.NormaliseAngle(self.phi_rotated)

        #precompute
        self.rgb = fn.ColourMap(self.norm_phi)
        self.masked = fn.ApplyMask(self.rgb, np.array(self.img), rgb_id = True)

    #function which processes and plots the skeleton images
    def processSkeleton(self):

        #radially dilates enpoints from PNG image
        self.dilated_endpoints = dl.dilate_endpoints_isotropic(self.endpoints_png, self.radius)
        
        #recombining the dilated endpoints with original binary skeleton
        self.combined_image = np.maximum(self.binary_Image, self.dilated_endpoints)
    
        #plotting
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes[0].imshow(self.binary_Image, cmap='gray')
        axes[0].set_title('Original PNG Skeleton')
        
        axes[1].imshow(self.processed_png, cmap='hot')
        axes[1].set_title("Processed PNG Skeleton")

        axes[2].imshow(self.highlighted_png)
        axes[2].set_title("Highlighted PNG Features")

        axes[3].imshow(self.combined_image, cmap='gray')
        axes[3].set_title(f'Dilated Endpoints (Radius={self.radius}) (PNG)')

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    #function to process a png image and produce colour diagram of angles and display
    def getOrientation(self, filename):

        #both items placed in np.array for processing
        #sl is the size of the gaussian kernal

        #creating colourmap
        colour_wheel, colour_wheel_transparent = fn.ColourWheel()
        

        # Plot the results for the PNG image
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))

        axes[0].imshow(self.img, cmap='gray')
        axes[0].set_title('Original Image')

        axes[1].imshow(np.array(self.processed_png))
        axes[1].set_title("Processed Image")

        axes[2].imshow(self.rgb)
        axes[2].set_title("Orientation Map")

        axes[3].imshow(self.masked)
        axes[3].set_title("Masked Orientation")

        axes[4].imshow(colour_wheel, extent=[-1, 1, -1, 1])
        axes[4].axis('off')

        """
        #square plotting for better presentation
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))  # 2x2 grid

        axes[0, 0].imshow(np.array(self.processed_png))
        axes[0, 0].set_title("Processed Image")

        axes[0, 1].imshow(rgb)
        axes[0, 1].set_title("Orientation Map")

        axes[1, 0].imshow(masked)
        axes[1, 0].set_title("Masked Orientation")

        axes[1, 1].imshow(colour_wheel, extent=[-1, 1, -1, 1])
        axes[1, 1].axis('off')

        """

        # Add angle labels around the wheel
        angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
        for angle_index in angles:
            x_text = 1.1 * np.cos(angle_index)
            y_text = 1.1 * np.sin(angle_index)
            angle_degrees = np.degrees(angle_index)
            plt.text(x_text, y_text, f"{int(angle_degrees)}°", ha='center', va='center')

        skel.SaveFigure(self.masked,filename, rgb=True)

        #show plots
        plt.show()

    #Function which will calculate the correlation of the orientations, makes use of a skeletonised image which can be processed in python or passed into the function
    #print statements have been commented out but left in for the sake of debugging
    def calcualteOrientationCorrelation(self, coarsening=0):
        """
        Compute the orientation correlation function for a nematic image.
    
        Parameters:
        img (numpy.ndarray): Binary mask image (1 for valid regions, 0 elsewhere).
        phi (numpy.ndarray): Orientation angle map (same shape as img).
        coarsening (int): Coarsening level (0 for all points, 1 for every other, etc.).
    
        Returns:
        distance_list (list): List of computed distances.
        correlation_list (list): List of corresponding dot product moduli.
        """

        #check for if skeletonised image has been passed
        if self.skeletonImagePath == None:
            skeletonImage=skeletonize(np.array(self.binary_Image))
            yx_indices = np.argwhere(np.array(skeletonImage) > 0)
            #print(len(yx_indices), 'len yx_indices skeletonCreated')
        else:
            # Get indices of masked pixels
            yx_indices = np.argwhere(np.array(self.skeletonImage) > 0)
            #print(len(yx_indices), 'len yx_indices skeletonpassed')
        
        # Apply coarsening only to the `i` loop: Select every (coarsening + 1)th point
        if coarsening > 0:
            yx_indices = yx_indices[::(coarsening + 1)]
        
        num_points = len(yx_indices)
        #print(num_points, 'num_points')
        
        #error checking
        if num_points < 2:
            raise ValueError("Not enough masked points to compute correlation.")
        
        # Compute pairwise distances and orientation correlations
        self.distance_list = []
        self.correlation_list = []
        
        for i in range(num_points):  # Reduced number of `i` due to coarsening
            #print(i)
            y1, x1 = yx_indices[i]
            angle1 = self.phi_rotated[y1, x1]
            
            v1 = np.array([np.cos(angle1), np.sin(angle1)])  # Nematic symmetry
            
            for y2, x2 in np.argwhere(np.array(self.processed_png) > 0):  # Loop over all masked pixels
                if (y1, x1) == (y2, x2):
                    self.distance_list.append(0)
                    self.correlation_list.append(1)
                
                angle2 = self.phi_rotated[y2, x2]
                v2 = np.array([np.cos(angle2), np.sin(angle2)])

                # Compute modulus of dot product
                dot_product = np.abs(np.dot(v1, v2))

                # Compute Euclidean distance
                distance = np.linalg.norm([y2 - y1, x2 - x1])

                self.distance_list.append(distance)
                self.correlation_list.append(dot_product**2)

        return self.distance_list, self.correlation_list
    
    #Binning function, requires bin size to be entered
    def binIt(self, bin_size):
        """
        Bin the distances and compute average correlation with standard error.
        
        Parameters:
        distance_list (list): List of computed distances.
        correlation_list (list): List of corresponding dot product moduli.
        bin_size (int): Bin size for averaging correlation as a function of distance.
        
        Returns:
        bin_centers (numpy.ndarray): Array of distance bin centers.
        correlation_avg (numpy.ndarray): Average modulus of the dot product at each distance bin.
        counts (numpy.ndarray): Number of contributions in each bin.
        std_err (numpy.ndarray): Standard error of the mean for each bin.
        """
        #calculating parameters
        max_distance = max(self.distance_list)
        bins = np.arange(0, max_distance + bin_size, bin_size)
        bin_centers = bins[:-1] + bin_size / 2
        
        correlation_avg = np.zeros(len(bin_centers))
        counts = np.zeros(len(bin_centers))
        std_dev = np.zeros(len(bin_centers))
        
        bin_values = [[] for _ in range(len(bin_centers))]
        
        for d, c in zip(self.distance_list, self.correlation_list):
            bin_index = int(d // bin_size)
            if bin_index < len(correlation_avg):
                bin_values[bin_index].append(c)
                correlation_avg[bin_index] += c
                counts[bin_index] += 1
        
        # Compute the averages and standard deviations
        for i in range(len(bin_centers)):
            if counts[i] > 0:
                correlation_avg[i] /= counts[i]
                std_dev[i] = np.std(bin_values[i]) if len(bin_values[i]) > 1 else 0
        
        # Compute standard error of the mean (SEM)
        std_err = np.zeros(len(bin_centers))
        std_err[counts > 0] = std_dev[counts > 0] / np.sqrt(counts[counts > 0])
        
        correlation_avg[counts == 0] = np.nan  # Avoid division by zero
        std_err[counts == 0] = np.nan  # Avoid division by zero
        
        return bin_centers, correlation_avg, counts, std_err

    #Used for plotting, should pass in correlation_avg which comes from function above, unsure of the form of the calculation
    def calculateCorrelationAvgNematic(self, correlationAvg):
        return (correlationAvg - 0.5)/(1-0.5)
    
    #plotting orientation correlation graph
    def plotOrientationCorrelation(self, bin_centers, correlation_avg_nematic, std_err, point_size, xlim, ylim, title, magnification):
        
        #in units of microns
        if magnification == 20:
            pixelSize = 0.3236
        elif magnification == 10:
            pixelSize = 0.651
        elif magnification == 4:
            pixelSize = 1.6169



        #convert from pixels to microns for plotting
        bin_centers = bin_centers*pixelSize

        plt.errorbar(bin_centers, correlation_avg_nematic, yerr=std_err, label='Orientation Correlation') #removed s=point_size, it doesnt like it

        #setting properties of graph
        #plt.xlim(xlim[0], xlim[1])
        plt.xlim(bin_centers[0], bin_centers[-1])
        plt.xlabel('Distance ($\mu m$)')
        plt.ylim(ylim[0], ylim[1])
        plt.ylabel('Correlation')
        plt.title(title)

        plt.show()

    #does all, and produces a graph, currently the axes limits are set to length of the lists, but could be changed if need be
    def produceCorrelationGraph(self, coarsening, title, magnification, bin_size=2, plotting=True):
        #Calculations
        distance_list, correlation_list = self.calcualteOrientationCorrelation(coarsening=coarsening) #is this line needed, I think it has already been calculated and can be omitted

        bin_centers, correlation_avg, counts, std_err = self.binIt(bin_size)

        correlationAvgNematic = self.calculateCorrelationAvgNematic(correlation_avg)

        #Plotting
        if plotting==True:
            self.plotOrientationCorrelation(bin_centers, correlationAvgNematic, std_err, point_size=2, xlim=[0,len(bin_centers)], ylim=[min(correlationAvgNematic), max(correlationAvgNematic)], title=title, magnification=magnification)

        #temporary
        return bin_centers, correlation_avg
    
    #produces curves
    def produceBinCenterGraphs(self, coarsening, bin_size=2):
        #Calculations
        distance_list, correlation_list = self.calcualteOrientationCorrelation(coarsening=coarsening)

        bin_centers, correlation_avg, counts, std_err = self.binIt(bin_size)

        #Plotting
        plt.figure(figsize=(10, 5))

        # Original skeleton
        plt.subplot(1, 2, 1)
        plt.scatter(bin_centers,correlation_avg)
        plt.title('Title 1')

        plt.subplot(1, 2, 2)
        plt.scatter(bin_centers,counts)
        plt.title('Title 2')

        plt.show()



"""
#code can handle if you enter a skeletonised image, or if you don't then it will produce one to use

a=ImageAnalysis('/Users/johnwhitfield/Desktop/projectSeagrass/Lab/Microscopy/agarComparison/0.5_10x_Ph1_Images/2025-07-08_Ph1_10x_edgesAndInsides_MMStack_Default.ome0004.jpg', None, 4, 4)
print(a.img)
#a.getOrientation('bananas')
a.produceCorrelationGraphData(10000, '0.5% Agar')
#a.produceBinCenterGraphs(10000)




#b=ImageAnalysis('/Users/johnwhitfield/Desktop/projectSeagrass/code/copyOfJoesCode/Demo/20xBF_1.7.png', None, 4, 4)
#b.getOrientation('bananas')
#b.produceCorrelationGraph(10000, 'testingNewSkeleton')

#d = ImageAnalysis('/Users/johnwhitfield/Desktop/projectSeagrass/code/copyOfJoesCode/Demo/20xBF_1.7.png', '/Users/johnwhitfield/Desktop/projectSeagrass/code/copyOfJoesCode/Demo/20xBF_1.7_Skel.png', 4, 4)
#d.produceCorrelationGraph(10000, 'testingPassingSkeleton')
#d.produceBinCenterGraphs(10000)

"""