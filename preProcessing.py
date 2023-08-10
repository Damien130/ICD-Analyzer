import numpy as np
import cv2
import cv2.cuda as cv2c # type: ignore
from PIL import Image, ImageSequence
from tqdm import tqdm
from skimage import color
from scipy import ndimage
from scipy.ndimage import binary_dilation
from io import BytesIO
import os



class preProcessor_counter:
    def __init__(self) -> None:
        pass
        
    def process_image(self, image_path, contrast_iterations
                      ,threshold = 0, HDD = False, image_data = None):
        
        # Create the background subtractor in GPU
        bg_subtractor = cv2c.createBackgroundSubtractorMOG2(history=400)
        #canny_edge_detector = cv2c.createCannyEdgeDetector(low_thresh=100, 
        #                                                   high_thresh=200, 
        #                                                   apperture_size=3, 
        #                                                  L2gradient=True)
        if HDD == False:
            image = Image.open(image_path)
        else :
            image = Image.open(BytesIO(image_data))

        num_frames = len(list(ImageSequence.Iterator(image)))  # Get the total number of frames
        bar_format = '{desc}: {percentage:3.0f}%\x1b[33m|{bar}\x1b[0m| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        progress_bar = tqdm(total=num_frames, desc="Preprocessing", leave=False, 
                            bar_format=bar_format)  # Initialize the progress bar

        single_pixel_lines = []  # Initialize an empty list to store single-pixel lines.
        for page in ImageSequence.Iterator(image):
            # convert the page to grayscale
            frame = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
            
            # Transfer the frame to the GPU memory
            gpu_frame = cv2c.GpuMat()
            gpu_frame.upload(frame)
            stream = cv2c.Stream_Null()

            # Enhance the contrast of the frame
            for i in range(contrast_iterations):
                gpu_frame = self.enhance_contrast(gpu_frame, stream=stream)

            # Apply background subtraction on the GPU
            fg_mask_gpu = bg_subtractor.apply(gpu_frame, learningRate=0.05, 
                                              stream = stream)

            # Download the foreground mask from GPU memory to CPU memory
            foreground_mask_cpu = cv2c.GpuMat.download(fg_mask_gpu)

            # Apply the foreground mask to the frame
            subtracted_frame = cv2.bitwise_and(frame, frame, mask=foreground_mask_cpu)

            gpu_frame.release()

            # Convert edges to binary
            subtracted_frame = subtracted_frame > threshold

            # fill holes
            subtracted_frame = ndimage.binary_fill_holes(subtracted_frame)

            # Label the connected components
            label_objects, _ = ndimage.label(subtracted_frame)
            sizes = np.bincount(label_objects.ravel())
            # greater than 7 microns, less than 25 microns
            mask_sizes = (sizes > 50) & (sizes < 500) 
            mask_sizes[0] = 0 # exclude background, which is labeled as 0
            subtracted_frame = mask_sizes[label_objects] # apply mask

            # Crop the processed frame to a single-pixel line from the middle.
            cropped_line = subtracted_frame[subtracted_frame.shape[0] // 2]
            # Reshape the 1D array to a 2D array with a second dimension of size 1
            cropped_line = np.expand_dims(cropped_line, axis=1)
            single_pixel_lines.append(cropped_line)
            progress_bar.update(1)  # Increment the progress bar

        progress_bar.close()  # Close the progress bar once the loop is done

        # Concatenate all the single-pixel lines horizontally to create the return image.
        return_image = np.concatenate(single_pixel_lines, axis=1)
        return_image = ndimage.binary_fill_holes(return_image)

        # Merge nearby white pixels using dilation
        return_image = binary_dilation(return_image, structure=np.ones((5, 5)))

        # release memory
        image.close()
        return return_image
    
    def process_image_demo(self, image_path, contrast_iterations, generate_image, output_path, threshold=0):
        # Create the background subtractor in GPU
        bg_subtractor = cv2c.createBackgroundSubtractorMOG2(history=400)
        #canny_edge_detector = cv2c.createCannyEdgeDetector(low_thresh=100,
        #                                                   high_thresh=200,
        #                                                   apperture_size=3,
        #                                                   L2gradient=True)

        image = Image.open(image_path)  # open the tif file
        num_frames = len(list(ImageSequence.Iterator(image)))  # Get the total number of frames
        bar_format = '{desc}: {percentage:3.0f}%\x1b[33m|{bar}\x1b[0m| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        progress_bar = tqdm(total=num_frames, desc="Preprocessing", leave=False, 
                            bar_format=bar_format)  # Initialize the progress bar

        # Initialize an empty list to store all the frames.
        all_frames = []
        if generate_image == True:
            single_pixel_lines = []  # Initialize an empty list to store single-pixel lines.

        for page in ImageSequence.Iterator(image):
            # convert the page to grayscale
            frame = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)

            # Output processed frames as images
            if generate_image == True:
                original_path = output_path + 'original/'
                grey_path = output_path + 'grey/'
                if not os.path.exists(original_path):
                    os.makedirs(original_path)
                if not os.path.exists(grey_path):
                    os.makedirs(grey_path)
                origin_frame = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)
                cv2.imwrite(original_path + str(progress_bar.n) + '.png', origin_frame)
                cv2.imwrite(grey_path + str(progress_bar.n) + '.png', frame)

            # Transfer the frame to the GPU memory
            gpu_frame = cv2c.GpuMat()
            gpu_frame.upload(frame)
            stream = cv2c.Stream_Null()

            # Enhance the contrast of the frame
            for i in range(contrast_iterations):
                gpu_frame = self.enhance_contrast(gpu_frame, stream=stream)
            
            if generate_image == True:
                contrast_path = output_path + 'contrast/'
                if not os.path.exists(contrast_path):
                    os.makedirs(contrast_path)
                # Download the frame from GPU memory to CPU memory
                gen_frame = cv2c.GpuMat.download(gpu_frame)
                cv2.imwrite(contrast_path + str(progress_bar.n) + '_contrast.png', gen_frame)

            # Apply background subtraction on the GPU
            fg_mask_gpu = bg_subtractor.apply(gpu_frame, learningRate=0.05,
                                              stream=stream)
            
            if generate_image == True:
                back_sub_path = output_path + 'fg_mask/'
                if not os.path.exists(back_sub_path):
                    os.makedirs(back_sub_path)
                # Download the foreground mask from GPU memory to CPU memory
                gen_frame = cv2c.GpuMat.download(fg_mask_gpu)
                cv2.imwrite(back_sub_path + str(progress_bar.n) + '_fg_mask.png', gen_frame)

            # Download the foreground mask from GPU memory to CPU memory
            foreground_mask_cpu = cv2c.GpuMat.download(fg_mask_gpu)

            # Combine the edges and foreground mask
            # combined_frame = np.logical_or(edges_binary, foreground_mask_cpu)

            # Apply the foreground mask to the frame
            subtracted_frame = cv2.bitwise_and(frame, frame, mask=foreground_mask_cpu)

            # Output processed frames as images
            if generate_image == True:
                subtracted_path = output_path + 'subtracted/'
                if not os.path.exists(subtracted_path):
                    os.makedirs(subtracted_path)
                cv2.imwrite(subtracted_path + str(progress_bar.n) + '_subtracted.png', subtracted_frame)

            gpu_frame.release()
            #gpu_frame.upload(subtracted_frame)

            # Apply Canny edge detection
            #edges_gpu = canny_edge_detector.detect(gpu_frame, stream=stream)

            # Download the edges from GPU memory to CPU memory
            #edges_cpu = cv2.cuda_GpuMat.download(edges_gpu)

            # Convert edges to binary
            subtracted_frame = subtracted_frame > threshold

            # Output processed frames as images
            if generate_image == True:
                threshold_path = output_path + 'threshold/'
                if not os.path.exists(threshold_path):
                    os.makedirs(threshold_path)
                cv2.imwrite(threshold_path + str(progress_bar.n) + '_threshold.png', subtracted_frame * 255)

            # fill holes
            subtracted_frame = ndimage.binary_fill_holes(subtracted_frame)

            # Output processed frames as images
            if generate_image == True:
                fill_holes_path = output_path + 'fill_holes/'
                if not os.path.exists(fill_holes_path):
                    os.makedirs(fill_holes_path)
                cv2.imwrite(fill_holes_path + str(progress_bar.n) + '_fill_holes.png', subtracted_frame * 255)
            # Label the connected components
            label_objects, _ = ndimage.label(subtracted_frame)
            sizes = np.bincount(label_objects.ravel())
            # greater than 7 microns, less than 25 microns
            mask_sizes = (sizes > 20) & (sizes < 200) 
            mask_sizes[0] = 0 # exclude background, which is labeled as 0
            subtracted_frame = mask_sizes[label_objects] # apply mask

            subtracted_frame = (subtracted_frame * 255).astype(np.uint8)

            if generate_image == True:
                label_path = output_path + 'label/'
                if not os.path.exists(label_path):
                    os.makedirs(label_path)
                cv2.imwrite(label_path + str(progress_bar.n) + '_label.png', subtracted_frame)
                # Crop the processed frame to a single-pixel line from the middle.
                cropped_line = subtracted_frame[subtracted_frame.shape[0] // 2]
                # Reshape the 1D array to a 2D array with a second dimension of size 1
                cropped_line = np.expand_dims(cropped_line, axis=1)
                line_path = output_path + 'line/'
                if not os.path.exists(line_path):
                    os.makedirs(line_path)
                cv2.imwrite(line_path + str(progress_bar.n) + '_line.png', cropped_line * 255)
                single_pixel_lines.append(cropped_line)

            all_frames.append(subtracted_frame)

            progress_bar.update(1)  # Increment the progress bar

        progress_bar.close()  # Close the progress bar once the loop is done

        if generate_image == True:
            unfilled_path = output_path + 'unfilled/'
            filled_path = output_path + 'filled/'
            dilated_path = output_path + 'dilated/'
            if not os.path.exists(unfilled_path):
                os.makedirs(unfilled_path)
            if not os.path.exists(filled_path):
                os.makedirs(filled_path)
            if not os.path.exists(dilated_path):
                os.makedirs(dilated_path)
            # Concatenate all the single-pixel lines horizontally to create the return image.
            return_image = np.concatenate(single_pixel_lines, axis=1)
            cv2.imwrite(unfilled_path + 'return_image_unfilled.png', return_image * 255)
            return_image = ndimage.binary_fill_holes(return_image)
            cv2.imwrite(filled_path + 'return_image_filled.png', return_image * 255)
            
            # Merge nearby white pixels using dilation
            return_image = binary_dilation(return_image, structure=np.ones((5, 5)))
            cv2.imwrite(dilated_path + 'return_image_dilated.png', return_image * 255)

        # release memory
        image.close()

        # Return the array of all frames
        return np.array(all_frames)

    def enhance_contrast(self, frame, stream=None):
        # Apply contrast enhancement using CLAHE 
        # (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2c.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_frame = clahe.apply(frame, stream = stream)

        return enhanced_frame

    def denoise(self, frame):
        # Denoise the frame using Non-local Means Denoising
        # parameters: h=10, hColor=10, templateWindowSize=7, searchWindowSize=35
        denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 30)
        return denoised_frame
    
class preProcessor_CV:
    def process_image(self, image_path, brightness_factor, contrast_iterations):
        # Create the background subtractor in GPU
        bg_subtractor = cv2c.createBackgroundSubtractorMOG2(history=250)
        image = Image.open(image_path)
        #background = Image.open(background_path).convert('RGB')
        #background = background.crop((125, 0, background.width - 125, background.height))

        frames = []
        for page in ImageSequence.Iterator(image):
            #cropped_page = page.crop((125, 0, page.width - 125, page.height))
            frame = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2BGR)

            # Transfer the frame to the GPU memory
            gpu_frame = cv2c.GpuMat(frame)
            stream = cv2c.Stream_Null()
            # Apply background subtraction on the GPU
            fg_mask_gpu = bg_subtractor.apply(gpu_frame, learningRate=0.1, 
                                              stream = stream)

            # Download the foreground mask from GPU memory to CPU memory
            foreground_mask_cpu = cv2c.GpuMat.download(fg_mask_gpu)

            # Apply the foreground mask to the frame
            subtracted_frame = cv2.bitwise_and(frame, frame, mask=foreground_mask_cpu)
            
            # Enhance the contrast of the frame
            for i in range(contrast_iterations):
                subtracted_frame = self.enhance_contrast(subtracted_frame)
            
            # Increase the brightness of the frame
            subtracted_frame = self.increase_brightness(subtracted_frame, brightness_factor)
            
            # Denoise the frame using Non-local Means Denoising
            subtracted_frame = self.denoise(subtracted_frame)

            frames.append(subtracted_frame)

        # release memory
        image.close()
        #background.close()
        return frames

    def increase_brightness(self, frame, brightness_factor):
        # Convert frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Scale the V channel (brightness) by the brightness_factor
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)

        # Convert back to BGR color space
        brightened_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        return brightened_frame


    def enhance_contrast(self, frame):
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

    def denoise(self, frame):
        # Denoise the frame using Non-local Means Denoising
        # parameters: h=10, hColor=10, templateWindowSize=7, searchWindowSize=35
        denoised_frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 30)
        return denoised_frame