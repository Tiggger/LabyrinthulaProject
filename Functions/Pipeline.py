
import os
import importlib
import numpy as np
from PIL import Image
from dataclasses import dataclass
from skimage.morphology import skeletonize, remove_small_holes, remove_small_objects

import Functions.ProcessSkeleton as skel
import Functions.Orientation as fn
import Functions.Dilation as dl

importlib.reload(fn)
importlib.reload(dl)
importlib.reload(skel)


@dataclass
class PipelineParameters:
    cell_hole: int
    cell_object: int
    track_hole: int
    track_object: int
    max_extension : int
    

class ExtensionPipeline:

    def __init__(self, 
                 output_parent_path, 
                 file_id, 
                 cell_path, 
                 track_path, 
                 parameters):
        
        self.output_parent_path = output_parent_path
        self.file_id = file_id
        self.cell_path = cell_path
        self.track_path = track_path
        self.params = parameters



    def output_directories(self):
        '''
        Create the output directories as needed
        '''

        # Define the directories
        self.dilated_dir = os.path.join(self.output_parent_path, "Dilated", self.file_id)
        self.sticky_dir = os.path.join(self.output_parent_path, "Sticky", self.file_id)

        # Create directories if they don't exist
        os.makedirs(self.dilated_dir, exist_ok=True)
        os.makedirs(self.sticky_dir, exist_ok=True)


    def load_images(self):
        '''
        Load the skeleton and cells and threshold
        '''
        # New code: Process a PNG image
        input_path_cell = self.cell_path  # Replace with PNG file path
        input_path_track = self.track_path  # Replace with PNG file path

        img_cell = np.array(Image.open(input_path_cell).convert('L'))  # Convert to grayscale
        img_track = np.array(Image.open(input_path_track).convert('L'))  # Convert to grayscale

        self.cell = np.array(img_cell) > 128  # Convert to binary (thresholding)
        self.track = np.array(img_track) > 128  # Convert to binary (thresholding)


    def prune_images(self):
        '''
        Prune images by removing small objects and filling holes

        '''

        # Remove Holes
        cell_filled = remove_small_holes(self.cell, area_threshold=self.params.cell_hole)
        track_filled = remove_small_holes(self.track, area_threshold=self.params.track_hole)

        # Prune Small Objects
        self.cell_pruned = remove_small_objects(cell_filled, min_size =
                                           self.params.cell_object, connectivity = 2)
        
        self.track_pruned = remove_small_objects(track_filled,
                                                 min_size = self.params.track_object, connectivity = 2)

        self.base = self.cell_pruned | self.track_pruned


    def orientation_map(self):
        '''
        Run the orientation module to establish the director map
        '''

        sl = 5  # Gaussian Kernel Size (See notes for sensible value)
        phi, nop = fn.OrientationFilter(self.track_pruned, sl)  # Pass skeleton to avoid conflict from cells
        # phi_rotated = fn.rotate_and_wrap_angles(phi,theta = np.pi/2)
        phi_new = fn.update_angles(phi,epsilon = 0.1,theta = np.pi/2)
        self.norm_phi = fn.NormaliseAngle(phi_new)


    def iterative_extension(self, radius, skeleton_iteration):
        '''
        Iterative loop for the extension of filaments
        '''

        skeleton_iteration = skeletonize(skeleton_iteration)

        # Combine using logical OR and rescale to 0 and 255
        skeleton_iteration = skeleton_iteration | self.cell_pruned

        # from the combination, make sure all endpoints remain endpoints in 8-connectivity sense by skeletonising again
        skeleton_iteration_endpoints = skeletonize(skeleton_iteration) # This will gnerate endpoints from the cells too which are later removed

        # Process the PNG skeleton
        _, _, endpoints = skel.process_skeleton(skeleton_iteration_endpoints.astype(int), A=10)
        # Find positions of the endpoints
        endpoint_positions = np.argwhere(endpoints)  # List of (row, col) coordinates of endpoints

        # Filter out endpoints that fall inside the blobs
        filtered_endpoints = np.array([pt for pt in endpoint_positions if not self.cell_pruned[pt[0], pt[1]]])

        dilated_skeleton = dl.iterate_dilation(filtered_endpoints,radius, self.norm_phi, skeleton_iteration)

        # Create a copy of the input skeleton
        input_skeleton_rgb = np.stack([skeleton_iteration] * 3, axis=-1).astype(float)  # Convert to RGB and float for matplotlib

        # Highlight the dilated regions in red
        dilated_additions = dilated_skeleton & ~self.base  # Pixels added by dilation
        dilated_skeleton_rgb = np.copy(input_skeleton_rgb)  # Copy the RGB base


        # Set the dilated additions to red (R=1, G=0, B=0)
        dilated_skeleton_rgb[dilated_additions] = [1, 0, 0]  # Red for new pixels

        # Find the new endpoints in the updated skeleton # TAG THIS IS REDUNDANT AND CAN BE REPLACED WITH FIND ENDPOINT NATIVE FUNCTION
        _, _, new_endpoints = skel.process_skeleton(dilated_skeleton, A=10)

        skeleton_iteration = dl.remove_unconnected_extensions_new(dilated_skeleton, new_endpoints, dilated_additions)

        # Combine using logical OR and rescale to 0 and 255
        skeleton_iteration_save = skeleton_iteration | self.cell_pruned

        # Save the figures
        skel.SaveFigure(dilated_skeleton_rgb, os.path.join(self.dilated_dir, f"dilated_{radius}.png"), rgb=True)
        skel.SaveFigure(skeleton_iteration_save, os.path.join(self.sticky_dir, f"sticky_{radius}.png"), rgb=False)

        return skeleton_iteration

    def run_all(self):

        self.output_directories()
        self.load_images()
        self.prune_images()
        self.orientation_map()

        skeleton_iteration = np.copy(self.track_pruned)
        
        for radius in range(0,self.params.max_extension):
            skeleton_iteration = self.iterative_extension(radius, skeleton_iteration)
        
