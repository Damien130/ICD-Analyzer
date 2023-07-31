import cv2
import imageio.v2 as imageio
import glob
import numpy as np
import os

# Specify the frame interval
frame_interval = 1

brightness_factor = 4.5
contrast_iterations = 5

# Specify the directory path containing the TIFF files
tiff_directory = '/mnt/d/Benam Lab/20230620/'
background_image_path = '/mnt/d/Benam Lab/20230620/background/background.tif'

# Specify the output video file path
output_video_path = '/mnt/h/testing/video.avi'

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

# Get the list of TIFF files in the directory
tiff_files = glob.glob(os.path.join(tiff_directory, '*.tif'))

# Open the first TIFF file to extract frame dimensions
first_frame = imageio.imread(tiff_files[0])
frame_shape = first_frame.shape

if len(frame_shape) == 3:
    frame_height, frame_width, _ = frame_shape
else:
    frame_height, frame_width = frame_shape

# Create a video writer object
output_video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 30.0, (frame_width, frame_height))

# Iterate over the TIFF files
for tiff_file_index, tiff_file in enumerate(tiff_files):
    # Read the TIFF file frames using imageio
    tiff_frames = imageio.imread(tiff_file, memtest=False)

    # Calculate the number of frames in the TIFF file
    num_frames = len(tiff_frames)

    # Determine the frame indices to extract
    frame_indices = range(0, num_frames, frame_interval)

    # Iterate over the frame indices
    for frame_index in frame_indices:
        # Get the current frame
        frame = tiff_frames[frame_index]

        # Crop the frame to remove the channel borders
        frame = frame[125:-125, :]

        background = imageio.imread(background_image_path)
        background = background[125:-125, :]

        # Perform background subtraction
        subtracted_frame = cv2.absdiff(frame, background)

        # Enhance contrast of subtracted frame
        for i in range(contrast_iterations):
            subtracted_frame = enhance_contrast(subtracted_frame)

        # Increase brightness of subtracted frame
        subtracted_frame = increase_brightness(subtracted_frame, brightness_factor)

        # Denoise the subtracted frame
        subtracted_frame = denoise(subtracted_frame)

        # Write the frame to the video writer
        output_video_writer.write(subtracted_frame)

        print(f"Processed frame {frame_index} from {tiff_file}")

# Release the video writer
output_video_writer.release()

print(f"Video saved: {output_video_path}")
