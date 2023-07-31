import os
import cv2
from PIL import Image, ImageSequence
from concurrent.futures import ThreadPoolExecutor
import numpy as np

background_image_path = '/mnt/d/Benam Lab/20230620/background/background.tif'
image_folder = '/mnt/d/Benam Lab/20230620' 
output_folder = '/mnt/h/testing/20230620'  # Specify the output directory
brightness_factor = 7
contrast_iterations = 6

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

    # Apply contrast enhancement using CLAHE (Contrast Limited Adaptive Histogram Equalization)
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

def process_image(image_path, background_path):
    image = Image.open(image_path)
    background = Image.open(background_path).convert('RGB')
    background = background.crop((125, 0, background.width - 125, background.height))

    frames = []
    for page in ImageSequence.Iterator(image):
        cropped_page = page.crop((125, 0, page.width - 125, page.height))
        frame = cv2.cvtColor(np.array(cropped_page), cv2.COLOR_RGB2BGR)
        background_array = cv2.cvtColor(np.array(background), cv2.COLOR_RGB2BGR)
        substracted_frame = cv2.subtract(frame, background_array)

        # Enhance the contrast of the frame
        for i in range(contrast_iterations):
            substracted_frame = enhance_contrast(substracted_frame)
        
        # Increase the brightness of the frame
        substracted_frame = increase_brightness(substracted_frame, brightness_factor)
        
        # Denoise the frame using Non-local Means Denoising
        #substracted_frame = denoise(substracted_frame)

        frames.append(substracted_frame)
    return frames

def convert_to_video(image_folder, output_folder, num_threads=32, videos_per_batch=32):
    tiff_images = sorted([file for file in os.listdir(image_folder) if file.endswith('.tif')], key=lambda x: int(x.split('_')[1].split('.')[0]))


    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create a ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        video_counter = 0
        frame_counter = 0
        video_writer = None

        for image_index, image_name in enumerate(tiff_images):
            # Process images in parallel using multiple threads
            image_path = os.path.join(image_folder, image_name)
            frames = process_image(image_path, background_image_path)
            
            if video_writer is None:
                frame_width, frame_height = frames[0].shape[1], frames[0].shape[0]
                #frame_width = 1190
                #frame_height = 200
                video_path = os.path.join(output_folder, f"video_{video_counter}.mov")
                video_writer = cv2.VideoWriter(video_path, 
                                               cv2.VideoWriter_fourcc
                                               (*"XVID"), 
                                               60, (frame_width, frame_height),
                                               6000)

            for frame in frames:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                video_writer.write(frame)
                frame_counter += 1

            # Check if a new video needs to be started
            if (image_index + 1) % videos_per_batch == 0:
                video_writer.release()
                video_writer = None
                video_counter += 1
                frame_counter = 0

            # Calculate progress
            progress = (image_index + 1) / len(tiff_images) * 100
            print(f"Progress: {progress:.2f}%")

        # Release the video writer for the last batch of frames
        if video_writer is not None:
            video_writer.release()


convert_to_video(image_folder, output_folder, num_threads=16, videos_per_batch=64)
