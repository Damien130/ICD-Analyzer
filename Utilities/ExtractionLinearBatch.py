import cv2
from PIL import Image
import glob
import random
import numpy as np
import os
import cv2
from concurrent.futures import ThreadPoolExecutor

# Specify the frame interval
frame_interval = 1

brightness_factor = 2
contrast_iterations = 2

# Specify the directory path containing the TIFF files
tiff_directory = '/mnt/i/20230710'
bg_subtractor = cv2.createBackgroundSubtractorMOG2()


def increase_brightness(frame, brightness_factor):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Scale the V channel (brightness) by the brightness_factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)

    # Convert back to BGR color space
    brightened_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return brightened_frame


def enhance_contrast(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply contrast enhancement using CLAHE 
    # (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Merge enhanced grayscale image with original frame
    enhanced_frame = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    enhanced_frame[:, :, :3] = frame

    return enhanced_frame

def denoise(frame):
    # Denoise the frame using Non-local Means Denoising
    # parameters: h=10, hColor=10, templateWindowSize=7, searchWindowSize=35
    denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 30)
    return denoised_frame

def extract_frame(tiff_file, output_directory, frame_index, tiff_file_index):
    
    # Open the TIFF file
    tiff = Image.open(tiff_file)

    # Go to the selected frame
    tiff.seek(frame_index)

    
    # Convert the frame to a NumPy array
    tiff = cv2.cvtColor(np.array(tiff), cv2.COLOR_RGB2BGR)
    
    # Apply background subtraction
    fg_mask = bg_subtractor.apply(frame)
    subtracted_frame = cv2.bitwise_and(tiff, tiff, mask=fg_mask)

    # Enhance contrast of subtracted frame
    for i in range(contrast_iterations):
        subtracted_frame = enhance_contrast(subtracted_frame)

    # Increase brightness of subtracted frame
    subtracted_frame = increase_brightness(subtracted_frame, brightness_factor)

    # Denoise the subtracted frame
    #subtracted_frame = denoise(subtracted_frame)

    # Save the frame as a PNG file
    output_path = os.path.join(output_directory, f"frame_{tiff_file_index}_{frame_index}.png")
    frame = Image.fromarray(subtracted_frame)
    frame.save(output_path, format="png", optimize=True, compress_level=9)

    print(f"Saved frame {frame_index} from {tiff_file}: {output_path}")

# Get the list of TIFF files in the directory
tiff_files = glob.glob(os.path.join(tiff_directory, '*.tif'))

# Create the output directory if it doesn't exist
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# Create a thread pool executor with the maximum number of worker threads
executor = ThreadPoolExecutor(max_workers=os.cpu_count())

# Iterate over the TIFF files
for tiff_file_index, tiff_file in enumerate(tiff_files):
    # Open the TIFF file
    tiff = Image.open(tiff_file)

    # Calculate the number of frames in the TIFF file
    num_frames = tiff.n_frames

    # Determine the frame indices to extract
    frame_indices = range(0, num_frames, frame_interval)

    # Submit the extract_frame function to the executor for each frame index
    for frame_index in frame_indices:
        executor.submit(extract_frame, tiff_file, output_directory, frame_index, tiff_file_index)

# Shutdown the executor and wait for all tasks to complete
executor.shutdown(wait=True)
