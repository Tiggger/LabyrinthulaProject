# Main file for orientation analysis functions using image moments

import numpy as np 
import matplotlib.cm as cm

from scipy.ndimage import convolve 
from matplotlib.colors import hsv_to_rgb 


from matplotlib.colors import hsv_to_rgb, Normalize 


# --------------------------------------
# ----------  Main Functions  ----------
# --------------------------------------

def OrientationFilter(img, sl):
    """
    Determines an orientation and nematic order parameter map of a raw image via various
    convolution operations with kernels. Returns the orientation φ and 
    strength of the nematic order parameter with same size as the original image `img`.
    `sl` specifies the size of the Gaussian kernel, i.e. 2*sl+1 in x- and y-direction 
    with standard deviation σ = sl/2.
    """

    ε = 1e-5  # small parameter for clipping

    # Define Gaussian kernel and coordinate grids for x and y
    sigma = sl / 2
    kernel_size = 2 * sl + 1
    #gc is matrix, calculated from outer product. Not entirely sure why it is exponential right now - form of gaussian?
    gc = np.outer(
        np.exp(-np.linspace(-sl, sl, kernel_size)**2 / (2 * sigma**2)),
        np.exp(-np.linspace(-sl, sl, kernel_size)**2 / (2 * sigma**2))
    )
    gc /= np.sum(gc)  # Normalize the Gaussian kernel

    # Define grids for weighted x and y coordinates
    gcpy = np.tile(np.arange(-sl, sl + 1), (kernel_size, 1)) #tile repeats first argument n times where second argument is n
    gcpx = gcpy.T

    # Convert image to grayscale and apply Gaussian-based filters
    nop_img = img.astype(np.float64)
    
    # Mean intensity with Gaussian kernel
    wt = np.clip(convolve(nop_img, gc), ε, np.inf) #clipping values which are not within epsilon and infinity region

    # Intensity-weighted x and y coordinates
    xw = convolve(nop_img, gc * gcpx)
    yw = convolve(nop_img, gc * gcpy)

    # Second moments (x^2, y^2, and xy)
    x2w = convolve(nop_img, gc * gcpx**2)
    y2w = convolve(nop_img, gc * gcpy**2)
    xyw = convolve(nop_img, gc * gcpx * gcpy)

    # Calculate the orientation angle φ
    phi = 0.5 * np.arctan2(-2 * (xyw * wt - xw * yw), x2w * wt - xw**2 - y2w * wt + yw**2)

    # Calculate the strength of the nematic order parameter (NOP)
    nop = np.sqrt(4 * (xyw * wt - xw * yw)**2 + (x2w * wt - xw**2 - y2w * wt + yw**2)**2)

    return phi, nop

def update_angles(phi,epsilon,theta):
    """
    Update angle values in the input array according to the given conditions:
      - For angles between 0 and pi/2: phi_new = pi/2 - phi
      - For angles between -pi/2 and 0: phi_new = -pi/2 - phi
      
    Parameters:
    phi_array (array-like): Array of angle values (in radians)
    
    Returns:
    numpy.ndarray: Updated array with modified angle values
    """
    # phi_array = np.array(phi_array)  # Ensure input is a NumPy array
    phi_new = phi.copy()       # Create a copy to store updated values
    
    # Apply transformations based on conditions
    mask1 = (phi > 0) & (phi < np.pi / 2)
    mask2 = (phi > -np.pi / 2) & (phi < 0)
    mask_special = (np.abs(phi) < epsilon) | (np.abs(np.abs(phi) - np.pi / 2) < epsilon)
    
    phi_new[mask1] = np.pi / 2 - phi[mask1]
    phi_new[mask2] = -np.pi / 2 - phi[mask2]
    phi_new[mask_special] = rotate_and_wrap_angles(phi[mask_special], theta)

    return phi_new

def rotate_and_wrap_angles(phi, theta=0):
    '''
    Function for modifying the output angles and insuring consistent domain, 
    this is actually not relevant when angles already restricted to -pi/2 to pi/2
    
    '''

    # Step 1: Rotate all angles in phi by theta (set to zero by default)
    rotated_phi = phi + theta

    # Step 2: Wrap angles to the range [-pi, pi]
    wrapped_phi = (rotated_phi + np.pi) % (2 * np.pi) - np.pi

    # Step 3: Adjust to the range [-pi/2, pi/2] symmetrically
    # If angle is in (pi/2, pi], map it to [-pi/2, 0]
    # If angle is in [-pi, -pi/2), map it to [0, pi/2]
    adjusted_phi = np.where(
        wrapped_phi > np.pi / 2,
        wrapped_phi - np.pi,
        np.where(
            wrapped_phi < -np.pi / 2,
            wrapped_phi + np.pi,
            wrapped_phi
        )
    )

    return adjusted_phi


def ApplyMask(rgb, img, rgb_id = True):
    # Ensure the mask is binary (0 and 1)
    binary_mask = np.where(img > 127, 1, 0)  # Thresholding to create a binary mask

    mask_rgb = binary_mask

    if rgb_id:
        # Expand the binary mask to have 3 channels (for RGB)
        mask_rgb = np.stack([binary_mask]*3, axis=-1)  # Shape will be (height, width, 3)

    # Apply the mask to the RGB color map
    masked_rgb = rgb * mask_rgb  # Keeps original color where mask is 1, sets to black where mask is 0

    return masked_rgb
    
def NormaliseAngle(phi):
    # Normalize angles from [-pi, pi] to [0, 1] over a range of pi (to reflect symmetry)
    # Create a piecewise normalization
    norm_phi = np.zeros_like(phi)
    mask1 = np.abs(phi) < (np.pi / 2)  # Condition for |angle| < π/2
    mask2 = phi >= np.pi / 2           # Condition for angle ≥ π/2
    mask3 = phi <= -np.pi / 2          # Condition for angle < -π/2

    # Apply piecewise normalization
    norm_phi[mask1] = 0.5 + phi[mask1]/np.pi           # For |angle| < π/2
    norm_phi[mask2] = -0.5 + phi[mask2]/np.pi          # For angle ≥ π/2
    norm_phi[mask3] = 1.5 + phi[mask3]/np.pi           # For angle < -π/2

    return norm_phi


def ColourMap(norm_phi, cmap_name='gist_rainbow'):
    """
    from the normalised function (angle values mapped to occupy range 0-1, apply the rgb colour map)
    """
    # Apply chosen color map
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(norm_phi)[..., :3]

    return rgb



def ColourWheel(inner_radius_fraction = 0.6, 
                wheel_size =     300, cmap_name = 'gist_rainbow'):
        
    cmap = cm.get_cmap(cmap_name)

    y, x = np.ogrid[-wheel_size:wheel_size, -wheel_size:wheel_size]
    r = np.sqrt(x**2 + y**2) / wheel_size  # Normalized radius
    angle = np.arctan2(y, x)  # Angle in radians from -pi to pi
    mask = (r <= 1) & (r >= inner_radius_fraction)  # Circular mask for the color wheel

    # Create a piecewise normalization for the color wheel
    norm_angle = NormaliseAngle(angle)

    # Generate the RGB wheel using the colormap
    rgb_wheel = cmap(norm_angle)
    rgb_wheel[~mask] = 1  # Set outside of the wheel to white

    # Generate the RGBA wheel using the colormap
    rgb_wheel_transparent = cmap(norm_angle)
    rgb_wheel_transparent[..., 3] = mask.astype(float)  # Set alpha channel to 1 within the mask, 0 outside

    return rgb_wheel, rgb_wheel_transparent
