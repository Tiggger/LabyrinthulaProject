import numpy as np
from scipy.ndimage import shift
from scipy.signal import correlate
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.color import rgb2gray

def compute_orientation_field(image, shift_distance=1, num_angles=8):
    """
    Compute the dominant orientation at each pixel using shift-based correlations.
    
    Args:
        image: 2D grayscale image (normalized to [0, 1]).
        shift_distance: Pixel distance for shifts (default: 1).
        num_angles: Number of angles to test (default: 8).
    
    Returns:
        orientation_field: 2D array of angles (in radians).
        correlation_strength: 2D array of max correlation values.
    """
    # Initialize outputs

    if len(image.shape) == 3:
        image=rgb2gray(image)

    height, width = image.shape
    orientation_field = np.zeros((height, width))
    correlation_strength = np.zeros((height, width))
    
    # Angles to test (0 to pi, excluding pi because it's redundant with 0)
    angles = np.linspace(0, np.pi, num_angles, endpoint=False)
    
    for y in range(height):
        print(y, 'y')
        for x in range(width):
            max_corr = -1   
            best_angle = 0
            
            for theta in angles:
                # Compute shift vector
                dx = shift_distance * np.cos(theta)
                dy = shift_distance * np.sin(theta)
                
                # Skip if shift goes outside image (or use boundary padding)
                if not (0 <= x + dx < width and 0 <= y + dy < height):
                    continue
                
                # Extract local patch (e.g., 3x3 around (x, y))
                patch = image[max(0, y-1):y+2, max(0, x-1):x+2]
                shifted_patch = shift(patch, (dy, dx), mode='reflect')
                
                # Compute normalized correlation
                numerator = np.sum(patch * shifted_patch)
                denominator = np.sum(patch ** 2)
                if denominator > 0:
                    corr = numerator / denominator
                else:
                    corr = 0
                
                # Track the best angle
                if corr > max_corr:
                    max_corr = corr
                    best_angle = theta
            
            orientation_field[y, x] = best_angle
            correlation_strength[y, x] = max_corr
    
    return orientation_field, correlation_strength

def calculate_nematic_tensor(orientation_field, correlation_strength, window_size=5):
    """
    Compute the nematic Q-tensor from the orientation field.
    
    Args:
        orientation_field: 2D array of angles (radians).
        correlation_strength: 2D array of correlation strengths.
        window_size: Size of averaging window (default: 5).
    
    Returns:
        S: 2D array of nematic order parameters.
        directors: 2D array of director vectors (angle in radians).
    """
    height, width = orientation_field.shape
    S = np.zeros((height, width))
    directors = np.zeros((height, width))
    
    # Pad arrays to handle boundaries
    pad = window_size // 2
    padded_angles = np.pad(orientation_field, pad, mode='reflect')
    padded_strength = np.pad(correlation_strength, pad, mode='reflect')
    
    for y in range(height):
        for x in range(width):
            # Extract local window
            angles_win = padded_angles[y:y+window_size, x:x+window_size]
            strength_win = padded_strength[y:y+window_size, x:x+window_size]
            
            # Compute Q-tensor components (weighted by correlation strength)
            cos2 = np.cos(2 * angles_win) * strength_win
            sin2 = np.sin(2 * angles_win) * strength_win
            
            Q_xx = np.mean(cos2)
            Q_xy = np.mean(sin2)
            
            # Compute eigenvalues/vectors
            eigenvalues = np.array([Q_xx + np.sqrt(Q_xx**2 + Q_xy**2), 
                                   Q_xx - np.sqrt(Q_xx**2 + Q_xy**2)])
            S[y, x] = np.max(eigenvalues)
            directors[y, x] = 0.5 * np.arctan2(Q_xy, Q_xx)
    
    return S, directors

# Example usage
if __name__ == "__main__":
    # Load and preprocess image (convert to grayscale, normalize)
    image_path = "/Users/johnwhitfield/Desktop/proper/t:20:21 - 2025-07-28_singleCellResolving+20xwithExtender_BF_.jpg"
    image = io.imread(image_path)
    if len(image.shape) == 3:
        image = color.rgb2gray(image)
    image = image / np.max(image)  # Normalize to [0, 1]
    
    # Compute orientation field
    orientation_field, correlation_strength = compute_orientation_field(image)
    
    # Compute nematic order (S) and directors
    S, directors = calculate_nematic_tensor(orientation_field, correlation_strength)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    
    axes[1].imshow(orientation_field, cmap='hsv')
    axes[1].set_title("Orientation Field (HSV)")
    
    # Plot directors (quiver plot)
    y, x = np.mgrid[:S.shape[0], :S.shape[1]]
    axes[2].imshow(S, cmap='viridis')
    axes[2].quiver(x[::5, ::5], y[::5, ::5], 
                   np.cos(directors[::5, ::5]), np.sin(directors[::5, ::5]), 
                   scale=20, color='red')
    axes[2].set_title("Nematic Order (S) and Directors")
    
    plt.tight_layout()
    plt.show()