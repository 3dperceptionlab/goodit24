import pandas as pd
import os
from PIL import Image

# Load the CSV file
df = pd.read_csv('/data/annotations.csv')

# Initialize lists to store the absolute widths and heights
absolute_widths = []
absolute_heights = []

# Directory where the images are stored
root_directory = '/data/images'

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Construct the filename
    filename = os.path.join(root_directory, row['video_id'], f"{row['video_id']}_{row['frame_id']:06d}.jpg")
    
    # Open the image and get its dimensions
    with Image.open(filename) as img:
        width, height = img.size
    
    # Convert relative coordinates to absolute pixel values
    xtl_pixels = row['xtl'] * width
    ytl_pixels = row['ytl'] * height
    xbr_pixels = row['xbr'] * width
    ybr_pixels = row['ybr'] * height

    # Calculate the width and height of the bounding box in pixels
    bbox_width = xbr_pixels - xtl_pixels
    bbox_height = ybr_pixels - ytl_pixels

    absolute_widths.append(bbox_width)
    absolute_heights.append(bbox_height)

# Calculate median width and height in pixels
median_width_pixels = pd.Series(absolute_widths).median()
median_height_pixels = pd.Series(absolute_heights).median()

print(f'Median Width in Pixels: {median_width_pixels}')
print(f'Median Height in Pixels: {median_height_pixels}')
