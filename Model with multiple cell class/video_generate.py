"""
This script generates an MP4 video from a sequence of image frames stored in the 'frames' folder.

Assumption:
- All frame image files are in PNG format.
- Each filename contains an integer that indicates its order in the sequence, e.g.:
    frame_1.png, frame_2.png, ..., frame_10.png
- No zero-padding is required (though it would also work with zero-padded filenames like frame_001.png).

The script ensures correct ascending order by extracting numbers from filenames before sorting.
"""

import imageio.v2 as imageio  # Use ImageIO v2 to avoid deprecation warnings
import os
import re

# Path to folder containing the frame images
frame_folder = "frames"

# Helper function to extract the numeric part from a filename
# For example, from "frame_12.png", it extracts 12
def extract_number(fname):
    match = re.search(r'\d+', fname)
    return int(match.group()) if match else -1  # If no number found, use -1

# List and numerically sort all .png files in the frame folder
frame_files = sorted([
    os.path.join(frame_folder, fname)
    for fname in os.listdir(frame_folder)
    if fname.endswith('.png')  # Only include PNG files
], key=lambda x: extract_number(os.path.basename(x)))  # Sort by extracted number

# Create a video writer using ffmpeg with 10 frames per second
with imageio.get_writer('simulation_recording.mp4', fps=15, format='ffmpeg') as writer:
    # Read and append each frame in sorted order
    for filename in frame_files:
        image = imageio.imread(filename)  # Load image from file
        writer.append_data(image)        # Write image to video
