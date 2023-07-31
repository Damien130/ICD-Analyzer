import os
import cv2
from tqdm import tqdm
from preProcessing import preProcessor_counter
import threading
import queue
import numpy as np

class binarySegmentationDemo:
    def __init__(self, folder_path, start_image_number=0, threshold=0, output_path = None, gen_images = False):
        self.folder_path = folder_path
        self.start_image_number = start_image_number
        self.threshold = threshold
        self.processor = preProcessor_counter()
        self.frame_queue = queue.Queue()
        self.output_path = output_path
        self.gen_images = gen_images

    def process_images_and_show_video(self):
        # Get a list of TIFF files in the folder
        tiff_files = [file for file in os.listdir(self.folder_path) if file.endswith('.tif')]
        tiff_files.sort()  # Sort files in ascending order

        # Define the processing thread
        def processing_thread():
            for tiff_file in tqdm(tiff_files, desc="Processing TIFF files", leave=False):
                tiff_file_path = os.path.join(self.folder_path, tiff_file)
                processed_frames = self.processor.process_image_demo(tiff_file_path, contrast_iterations=1, generate_image=self.gen_images, output_path=self.output_path, threshold=self.threshold)
                self.frame_queue.put(processed_frames)
                if self.gen_images is True:
                    break
            self.frame_queue.put(None) # Put None to the queue to signal the end of processing

        # Define the display thread
        def display_thread():
            first_frame = True
            window_name = 'Processed Frames'

            # define window size
            window_width = 1440
            window_height = 200
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, window_width, window_height)

            while True:
                frame_array = self.frame_queue.get()
                if frame_array is None:
                    break

                for frame in frame_array:
                    if not first_frame:
                        cv2.imshow(window_name, frame)
                    first_frame = False

                    # Wait for a key press event and close the window if 'Esc' key is pressed
                    key = cv2.waitKey(30)
                    if key == 27:  # 27 is the ASCII code for 'Esc' key
                        break

        # Start the processing and display threads
        processing_thread = threading.Thread(target=processing_thread)
        if self.gen_images is False:
            display_thread = threading.Thread(target=display_thread)
            display_thread.start()

        processing_thread.start()
        

        # Wait for both threads to finish
        processing_thread.join()
        if self.gen_images is False:
            display_thread.join()

        cv2.destroyAllWindows()

# Usage example:
#folder_path = '/path/to/your/folder'
#start_image_number = 0  # You can adjust this as needed
#image_processor = ImageProcessor(folder_path, start_image_number)
#image_processor.process_images_and_show_video()