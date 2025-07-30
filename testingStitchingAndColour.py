import os
from PIL import Image, ImageSequence, ImageDraw, ImageFont
import re
import copy
import numpy as np

# Set directories and parameters
image_dir = '/Users/johnwhitfield/Desktop/input'
filename_prefix = '2025-07-14_10x_BF_tile_1'
output_dir = '/Users/johnwhitfield/Desktop/output'

rows, cols = 4, 5
overlap = 0.1
max_timepoints = 1

def calculate_percentage(image):
    pixels = np.array(image)
    count0 = np.sum(pixels == 0)
    count1 = np.sum(pixels == 1)
    total = count0 + count1
    return (count0 / total) if total > 0 else 0

def get_percentage_color(percentage):
    # Blue (low %) → Yellow (high %)
    blue = int(255 * (1 - percentage))
    red = int(255 * percentage)
    green = int(255 * percentage)
    return (red, green, blue, 180)  # RGBA with transparency

def add_scale_bar(image, position='right'):
    """Adds a vertical color scale bar to the image with properly scaled text"""
    # Calculate dimensions based on image size
    base_size = max(image.width, image.height)
    scale_width = int(base_size * 0.015)  # 1.5% of largest dimension
    scale_height = int(base_size * 0.75)  # 25% of largest dimension
    margin = int(base_size * 0.1)  # 10% margin
    text_margin = int(scale_width * 0.5)  # Space between bar and text
    
    # Font sizes as percentage of image size
    title_font_size = int(base_size * 25)  # 0.8% of base size
    label_font_size = int(base_size * 25)  # 0.7% of base size
    
    # Create a new image with space for the scale bar
    if position in ['right', 'left']:
        new_width = image.width + scale_width + margin + text_margin * 8
        new_height = image.height
    else:
        new_width = image.width
        new_height = image.height + scale_height + margin
    
    new_image = Image.new("RGBA", (new_width, new_height), (255, 255, 255, 0))
    
    # Paste original image
    if position == 'right':
        new_image.paste(image, (0, 0))
    elif position == 'left':
        new_image.paste(image, (scale_width + margin + text_margin * 8, 0))
    else:
        new_image.paste(image, (0, 0))
    
    # Create scale bar with space for text
    scale_bar = Image.new("RGBA", (scale_width + text_margin * 8, 
                                 scale_height + int(scale_height * 0.15)), 
                        (255, 255, 255, 0))
    draw = ImageDraw.Draw(scale_bar)
    
    # Draw gradient (leaving space at top for title)
    bar_top = int(scale_height * 0.15)  # 15% space at top for title
    for y in range(bar_top, scale_height):
        percentage = 1 - ((y - bar_top) / (scale_height - bar_top))
        color = get_percentage_color(percentage)
        draw.line([(0, y), (scale_width, y)], fill=color)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", title_font_size)
        label_font = ImageFont.truetype("arial.ttf", label_font_size)
    except:
        # Fallback if font not found
        title_font = ImageFont.load_default()
        title_font.size = title_font_size
        label_font = ImageFont.load_default()
        label_font.size = label_font_size
    
    # Draw title
    title = "Cell Density (%)"
    title_width = title_font.getlength(title)
    draw.text(((scale_width - title_width)/2, 5), title, fill="black", font=title_font)
    
    # Add percentage markers with clear labels
    labels = [
        (0, bar_top, "0% (No cells)"),
        (25, bar_top + (scale_height - bar_top)*0.25, "25%"),
        (50, bar_top + (scale_height - bar_top)*0.5, "50%"),
        (75, bar_top + (scale_height - bar_top)*0.75, "75%"),
        (100, scale_height - 10, "100% (Full cells)")
    ]
    
    for val, pos, label in labels:
        # Calculate text position to avoid overflow
        text_x = scale_width + text_margin
        text_y = pos - label_font_size//2
        draw.text((text_x, text_y), label, fill="black", font=label_font)
    
    # Position scale bar
    if position == 'right':
        new_image.paste(scale_bar, (image.width + margin, 
                                  (image.height - scale_height)//2), scale_bar)
    elif position == 'left':
        new_image.paste(scale_bar, (0, (image.height - scale_height)//2), scale_bar)
    else:
        new_image.paste(scale_bar, ((image.width - scale_width)//2, 
                                  image.height + margin), scale_bar)
    
    return new_image

# Load images
images_grid = {}
for file in [f for f in os.listdir(image_dir) if f.endswith('.tif')]:
    match = re.search(r'Pos(\d{3})_(\d{3})', file)
    if match:
        x, y = match.groups()
        y = rows - 1 - int(y)  # Invert Y-axis
        img_stack = Image.open(os.path.join(image_dir, file))
        images_grid[(int(x), y)] = [copy.deepcopy(f) for f in ImageSequence.Iterator(img_stack)]

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

for t in range(min(max_timepoints, len(next(iter(images_grid.values()))))):
    # Create canvases
    sample_frame = next(iter(images_grid.values()))[0]
    stitched_width = int((cols - 1) * (1 - overlap) * sample_frame.size[0] + sample_frame.size[0])
    stitched_height = int((rows - 1) * (1 - overlap) * sample_frame.size[1] + sample_frame.size[1])
    
    stitched_16bit = Image.new("I", (stitched_width, stitched_height))
    stitched_color_overlay = Image.new("RGBA", (stitched_width, stitched_height))
    stitched_bw_overlay = Image.new("RGBA", (stitched_width, stitched_height))

    for (x, y), stack in images_grid.items():
        try:
            frame = stack[t]
        except IndexError:
            continue
        
        # Calculate % and color
        perc = calculate_percentage(frame)
        color = get_percentage_color(perc)
        print(f"Tile (X={x}, Y={y}): {perc * 100:.2f}% cells → Color: {color}")
        
        # Position calculation
        x_pos = (cols - 1 - x) * int(frame.size[0] * (1 - overlap))
        y_pos = y * int(frame.size[1] * (1 - overlap))
        
        # Process for COLOR background with overlay
        frame_rgba = frame.convert("RGBA")
        overlay = Image.new("RGBA", frame.size, color)
        frame_color_overlay = Image.alpha_composite(frame_rgba, overlay)
        stitched_color_overlay.paste(frame_color_overlay, (x_pos, y_pos), frame_color_overlay)
        
        # Process for BW background with overlay
        frame_bw = frame.convert("L").convert("RGBA")
        frame_bw_overlay = Image.alpha_composite(frame_bw, overlay)
        stitched_bw_overlay.paste(frame_bw_overlay, (x_pos, y_pos), frame_bw_overlay)
        
        # Paste original into 16-bit image
        stitched_16bit.paste(frame, (x_pos, y_pos))
    
    # Save outputs
    stitched_16bit.save(os.path.join(output_dir, f"{filename_prefix}_timepoint_{t}_16bit.tif"), compression="tiff_deflate")
    
    # Add scale bars and save overlay versions
    color_with_scale = add_scale_bar(stitched_color_overlay)
    color_with_scale.convert("RGB").save(os.path.join(output_dir, f"{filename_prefix}_timepoint_{t}_color_overlay.png"))
    print(f"Saved color overlay: {filename_prefix}_timepoint_{t}_color_overlay.png")
    
    bw_with_scale = add_scale_bar(stitched_bw_overlay)
    bw_with_scale.convert("RGB").save(os.path.join(output_dir, f"{filename_prefix}_timepoint_{t}_bw_overlay.png"))
    print(f"Saved BW overlay: {filename_prefix}_timepoint_{t}_bw_overlay.png")