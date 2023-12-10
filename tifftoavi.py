import cv2
import os
import glob
from PIL import Image
from tqdm import tqdm
import numpy as np

# Specify the directory where the TIFF files are located
directory = '/mnt/d/Benam/ICDExample/Example/Small'

# Get a list of all TIFF files in the directory
files = glob.glob(os.path.join(directory, '*.tif'))
files.sort()

# Read the first frame of the first TIFF file to get the shape
tiff = Image.open(files[0])
tiff.seek(0)
frame = np.array(tiff)
height, width, layers = frame.shape

# Create a VideoWriter object
output_file = os.path.join(directory, 'output.avi')
video = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'X264'), 60, (width, height))

# Initialize the progress bar
pbar = tqdm(total=len(files))

# Loop through the files
for file in files:
    tiff = Image.open(file)
    for i in range(tiff.n_frames):
        tiff.seek(i)
        frame = cv2.cvtColor(np.array(tiff), cv2.COLOR_RGB2BGR)
        video.write(frame)

    # Update the progress bar
    pbar.update(1)

# Release the VideoWriter object
video.release()

# Close the progress bar
pbar.close()