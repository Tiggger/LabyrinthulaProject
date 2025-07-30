import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
from tempfile import NamedTemporaryFile
from skimage.morphology import skeletonize
from functools import partial

class OptimizedImageAnalysis:
    def __init__(self, imagePath, skeletonImagePath=None, radius=4, sl=4):
        self.imagePath = imagePath
        self.radius = radius
        self.sl = sl
        
        # Load and preprocess image
        self.img = np.array(Image.open(imagePath).convert("L"))
        self.binary_Image = self.img > 128
        
        # Process skeleton
        if skeletonImagePath:
            self.skeletonImage = np.array(Image.open(skeletonImagePath).convert("L"))
            self.processed_png = (self.skeletonImage > 0).astype(int)
        else:
            self.processed_png = skeletonize(self.binary_Image)
        
        # Precompute orientation data
        self.phi = self._compute_orientation(self.processed_png)
        self.phi_rotated = self._rotate_angles(self.phi)
        
        # Precompute skeleton indices
        self.skeleton_indices = np.argwhere(self.processed_png > 0)
        
        # Initialize cache
        self._correlation_cache = {}

    def _compute_orientation(self, skeleton):
        # Replace with your actual orientation calculation
        height, width = skeleton.shape
        return np.random.rand(height, width) * np.pi  # Dummy data

    def _rotate_angles(self, phi):
        return (phi + np.pi/2) % np.pi

    def _vectorized_correlation(self, coarsening=0):
        yx_indices = self.skeleton_indices[::(coarsening + 1)] if coarsening > 0 else self.skeleton_indices
        
        # Precompute all vectors
        angles = self.phi_rotated[yx_indices[:, 0], yx_indices[:, 1]]
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        
        distance_list = []
        correlation_list = []
        
        for i in tqdm(range(len(yx_indices)), desc="Computing correlations"):
            y1, x1 = yx_indices[i]
            v1 = vectors[i]
            
            # Vectorized calculations
            dy = self.skeleton_indices[:, 0] - y1
            dx = self.skeleton_indices[:, 1] - x1
            distances = np.sqrt(dx**2 + dy**2)
            dots = np.abs(np.dot(vectors, v1))
            
            # Exclude self-correlation
            mask = distances > 0
            distance_list.extend(distances[mask])
            correlation_list.extend((dots[mask] ** 2))
        
        return distance_list, correlation_list

    def _process_chunk(self, chunk_indices, skeleton_indices, phi_rotated):
        chunk_dist = []
        chunk_corr = []
        
        # Precompute vectors for the chunk
        angles = phi_rotated[chunk_indices[:, 0], chunk_indices[:, 1]]
        vectors = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Precompute vectors for all skeleton points
        all_angles = phi_rotated[skeleton_indices[:, 0], skeleton_indices[:, 1]]
        all_vectors = np.column_stack([np.cos(all_angles), np.sin(all_angles)])
        
        for i in range(len(chunk_indices)):
            y1, x1 = chunk_indices[i]
            v1 = vectors[i]
            
            # Calculate distances and dot products
            dy = skeleton_indices[:, 0] - y1
            dx = skeleton_indices[:, 1] - x1
            distances = np.sqrt(dx**2 + dy**2)
            dots = np.abs(np.dot(all_vectors, v1))
            
            # Exclude self-correlation
            mask = distances > 0
            chunk_dist.extend(distances[mask])
            chunk_corr.extend((dots[mask] ** 2))
        
        return chunk_dist, chunk_corr

    def _parallel_correlation(self, coarsening=0):
        yx_indices = self.skeleton_indices[::(coarsening + 1)] if coarsening > 0 else self.skeleton_indices
        
        # Prepare data for multiprocessing
        chunk_size = max(1, len(yx_indices) // (cpu_count() * 2))
        chunks = [yx_indices[i:i + chunk_size] for i in range(0, len(yx_indices), chunk_size)]
        
        # Create partial function with necessary data
        process_func = partial(
            self._process_chunk,
            skeleton_indices=self.skeleton_indices,
            phi_rotated=self.phi_rotated
        )
        
        # Process chunks in parallel
        with Pool() as pool:
            results = []
            for result in pool.imap_unordered(process_func, chunks):
                results.append(result)
        
        # Combine results
        distance_list = []
        correlation_list = []
        for d, c in results:
            distance_list.extend(d)
            correlation_list.extend(c)
        
        return distance_list, correlation_list

    def calculate_correlation(self, coarsening=0, use_parallel=True):
        cache_key = f"corr_{coarsening}"
        if cache_key in self._correlation_cache:
            return self._correlation_cache[cache_key]
        
        if use_parallel and len(self.skeleton_indices) > 1000:
            dist, corr = self._parallel_correlation(coarsening)
        else:
            dist, corr = self._vectorized_correlation(coarsening)
        
        self._correlation_cache[cache_key] = (dist, corr)
        return dist, corr

    def bin_correlations(self, distance_list, correlation_list, bin_size=2):
        if not distance_list:
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        max_dist = max(distance_list)
        bins = np.arange(0, max_dist + bin_size, bin_size)
        bin_indices = np.digitize(distance_list, bins) - 1
        
        valid = (bin_indices >= 0) & (bin_indices < len(bins) - 1)
        bin_sums = np.bincount(bin_indices[valid], weights=np.array(correlation_list)[valid], 
                              minlength=len(bins)-1)
        bin_counts = np.bincount(bin_indices[valid], minlength=len(bins)-1)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            correlation_avg = np.where(bin_counts > 0, bin_sums / bin_counts, np.nan)
        
        # Calculate standard error
        std_err = np.zeros_like(correlation_avg)
        for i in range(len(bins)-1):
            mask = bin_indices == i
            if np.sum(mask) > 1:
                std_err[i] = np.std(np.array(correlation_list)[mask]) / np.sqrt(np.sum(mask))
        
        bin_centers = bins[:-1] + bin_size / 2
        return bin_centers, correlation_avg, bin_counts, std_err

    def produce_correlation_graph(self, coarsening=1000, title="Correlation", bin_size=2):
        # Dynamic coarsening adjustment
        img_size = self.img.shape[0] * self.img.shape[1]
        effective_coarsening = max(1, int(coarsening * (10000 / img_size)))
        
        distance_list, correlation_list = self.calculate_correlation(effective_coarsening)
        bin_centers, correlation_avg, counts, std_err = self.bin_correlations(
            distance_list, correlation_list, bin_size)
        
        # Normalize correlation
        correlation_nematic = (correlation_avg - 0.5) / (1 - 0.5)
        
        # Plotting
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.img, cmap='gray')
        plt.title("Cell Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.errorbar(bin_centers, correlation_nematic, yerr=std_err, 
                    fmt='-o', markersize=4, capsize=3)
        plt.xlabel('Distance (pixels)')
        plt.ylabel('Correlation')
        plt.title(title)
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def create_interactive_heatmap(image, cells, densities, cmap='viridis', alpha=0.5):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(image, cmap='gray')
    
    cell_height, cell_width = cells[0][0].shape
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    # Create clickable patches
    for y in range(len(cells)):
        for x in range(len(cells[0])):
            rect = patches.Rectangle(
                (x * cell_width, y * cell_height), cell_width, cell_height,
                linewidth=0, facecolor=sm.to_rgba(densities[y][x]), alpha=alpha
            )
            ax.add_patch(rect)
    
    plt.colorbar(sm, ax=ax, label='Density')
    
    def onclick(event):
        if event.inaxes != ax:
            return
            
        x, y = int(event.xdata), int(event.ydata)
        cell_x, cell_y = x // cell_width, y // cell_height
        
        if cell_y >= len(cells) or cell_x >= len(cells[0]):
            return
            
        clicked_cell = cells[cell_y][cell_x]
        
        # Use temp file to avoid memory issues
        with NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            Image.fromarray(clicked_cell).save(tmp.name)
            try:
                analyzer = OptimizedImageAnalysis(tmp.name, radius=4, sl=4)
                analyzer.produce_correlation_graph(
                    coarsening=1000,  # Smaller coarsening for cells
                    title=f"Cell ({cell_x}, {cell_y}) Correlation"
                )
            finally:
                try:
                    os.unlink(tmp.name)
                except:
                    pass
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.title("Click on any cell to analyze")
    plt.axis('off')
    plt.show()

# Utility functions
def split_into_cells(img, xsplit, ysplit):
    rows, cols = img.shape
    cell_height = rows // ysplit
    cell_width = cols // xsplit
    return [
        [
            img[y*cell_height:(y+1)*cell_height, x*cell_width:(x+1)*cell_width]
            for x in range(xsplit)
        ]
        for y in range(ysplit)
    ]

def calculate_density(cells):
    return [
        [
            np.sum(cell == 76) / (np.sum(cell == 76) + np.sum(cell == 188) + 1e-10)
            for cell in row
        ]
        for row in cells
    ]

# Example usage
if __name__ == "__main__":
    image_path = "/Users/johnwhitfield/Desktop/output/2025-07-21_Ph1_10x_smallTiles_1_timepoint_0.tif"
    img = np.array(Image.open(image_path))
    
    xsplit, ysplit = 16, 16
    cells = split_into_cells(img, xsplit, ysplit)
    densities = calculate_density(cells)
    
    create_interactive_heatmap(img, cells, densities)