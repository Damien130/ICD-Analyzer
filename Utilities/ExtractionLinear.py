import cv2
from PIL import Image
import glob
import random
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# Specify the frame interval and the TIFF file number
frame_interval = 100
tiff_number = 4

brightness_factor = 5
contrast_iterations = 5

# Specify the path to the specific TIFF file
tiff_file = f'/mnt/d/Benam Lab/20230620/LTB4-500nM-2.5uL-1_{tiff_number}.tif'
background_image_path = '/mnt/d/Benam Lab/20230620/background/background.tif'

# Specify the output directory
output_directory = '/mnt/h/training/png/batch5'

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

def extract_frame(tiff_file, output_directory, frame_index):
    # Open the TIFF file
    tiff = Image.open(tiff_file)
    background = Image.open(background_image_path).convert('RGB')

    # Go to the selected frame
    tiff.seek(frame_index)


    # Crop the frame to remove the channel borders
    tiff = tiff.crop((125, 0, tiff.width - 125, tiff.height))
    background = background.crop((125, 0, background.width - 125,
                                   background.height))
    
    # Convert the frame to a NumPy array
    tiff = cv2.cvtColor(np.array(tiff), cv2.COLOR_RGB2BGR)
    background = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
    
    # Perform background subtraction
    subtracted_frame = cv2.absdiff(tiff, background)

    # Enhance contrast of subtracted frame
    for i in range(contrast_iterations):
        substracted_frame = enhance_contrast(subtracted_frame)

    # Increase brightness of subtracted frame
    subtracted_frame = increase_brightness(subtracted_frame, brightness_factor)

    # Denoise subtracted frame
    #subtracted_frame = denoise(subtracted_frame)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the frame as a PNG file
    output_path = os.path.join(output_directory, f"frame_{tiff_number}_{frame_index}.png")
    frame = Image.fromarray(subtracted_frame)
    frame.save(output_path, format="png", optimize = True, 
               compress_level=9)

    print(f"Saved frame {frame_index} from {tiff_file}: {output_path}")



# Open the TIFF file
tiff = Image.open(tiff_file)

# Calculate the number of frames in the TIFF file
num_frames = tiff.n_frames

# Determine the frame indices to extract
frame_indices = range(0, num_frames, frame_interval)

for frame_index in frame_indices:
    extract_frame(tiff_file, output_directory, frame_index)
