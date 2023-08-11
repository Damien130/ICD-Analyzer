import cv2
import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from preProcessing import preProcessor_counter
from preProcessing import preProcessor_CV
import csv
import queue
from skimage import  measure
import curses
import time
from math import ceil  # Import the ceil function
import plotly.graph_objects as go
import psutil

class segmentation_ObjectCounter:
    def __init__(self) -> None:
        pass

    def predict_file(self, image_folder, save_dir, 
                     batch_size, buffer_size, threshold, debug=False,
                     lower_bound = 40, upper_bound = 500):
        
        tiff_images = sorted([file for file in os.listdir(image_folder) 
                             if file.endswith('.tif')], 
                             key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        image_dir = os.path.join(save_dir, 'images')
        
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)
        
        csv_file = os.path.join(save_dir, 'cell_count.csv')

        image_index = 0 # Initialize image index
        totalCount = 0 # Initialize total count
        frame_queue = queue.Queue()
        futures = OrderedDict()
        ppcs = preProcessor_counter() # Initialize the preprocessor object

        
        stdscr = curses.initscr() # Initialize the curses screen
        curses.start_color() # Start color manipulation

        # Calculate the total number of batches based on the total number of images and the batch size
        total_batches = ceil(len(tiff_images) / batch_size)
        # Initialize the progress bar
        stdscr.move(0, 0)
        progress_bar = tqdm(total=total_batches, desc='Overall Progress', unit='batch', leave=True)
        progress_bar.set_postfix({'Total Count': totalCount})
        
        # Initialize lists to store batch numbers and total counts for plotting
        batch_times = []
        total_counts = []

        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                expected_image_index = 0  # Initialize the expected image index
                batch_number = 0  # Initialize the batch number

                while expected_image_index < len(tiff_images):
                    
                    # Refill buffer if required
                    if len(futures) < buffer_size and image_index < len(tiff_images):
                        image_name = tiff_images[image_index]
                        image_path = os.path.join(image_folder, image_name)

                        # process_image(image_path, brighness factor, contrast iterations)
                        stdscr.move(1, 0) # move out of the progress bar's position
                        stdscr.clrtoeol()  # Clear the entire line
                        future = executor.submit(ppcs.process_image, image_path, 1, threshold=threshold,
                                                 lower_bound=lower_bound, upper_bound=upper_bound)
                        futures[image_index] = (image_name, future)
                        image_index += 1
                    
                    # Check if the expected future has completed
                    if expected_image_index in futures and futures[expected_image_index][1].done():
                        stdscr.move(0, 0)
                        stdscr.clrtoeol()  # Clear the entire line
                        stdscr.addstr(0, 0, str(progress_bar))
                        stdscr.refresh()
                        name, future = futures.pop(expected_image_index)

                        # Get the processed image
                        processed_image = future.result()

                        # Add the processed image to the queue
                        frame_queue.put((name, processed_image))

                        if debug == True:
                            frame_path = os.path.join(image_dir, name)
                            cv2.imwrite(frame_path, processed_image*255)
                        
                        expected_image_index += 1  # Move to the next expected image index

                        # Check if the batch is complete, or if the last image has been processed
                        if expected_image_index % batch_size == 0 or expected_image_index == len(tiff_images) - 1:
                            frame_queue.put(None) # indicate end of batch
                            # process the queue
                            totalCount = self.process_queue(frame_queue, 
                                                            totalCount, 
                                                            csv_file)
                            batch_number += 1
                            # Update the progress bar
                            progress_bar.set_postfix({'Total Count': totalCount})
                            progress_bar.update(1)

                            # Update the lists for plotting
                            time_elapsed = batch_number * batch_size * 1240 / 432000
                            batch_times.append(time_elapsed)
                            total_counts.append(totalCount)
                            # Plot the line chart
                            self.plot_line_chart(batch_times, total_counts, save_dir)
                        
                    else:
                        time.sleep(0.1) # Wait for the order to complete

                    # Move the progress bar to the top of the console
                    stdscr.move(0, 0)
                    stdscr.clrtoeol() # Clear the entire line
                    stdscr.addstr(0, 0, str(progress_bar))
                    # Refresh the screen to update the progress bar's position
                    stdscr.refresh()

                    
        finally:
            # Restore the terminal settings
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.curs_set(1)
            curses.endwin()

    def process_queue(self, queue, totalCount, csv_file):
        processed_images = 0

        # Iterate over the queue
        while True:
            image = queue.get()
            if image is None:
                break

            # Get the image name and the processed image
            image_name, processed_image = image

            # Label the binary image
            labeled_image = measure.label(processed_image)

            # Get the number of objects
            object_count = np.max(labeled_image)

            # Add the object count to the total count
            totalCount += object_count

            # write to csv file
            with open(csv_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([image_name, object_count, totalCount])
            
            processed_images += 1
        return totalCount
    
    def plot_line_chart(self, batch_times, total_counts, save_dir):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=batch_times, y=total_counts, mode='markers+lines'))

        fig.update_layout(
            xaxis_title='Time (hours)',
            yaxis_title='Total Count',
            title='Total Count vs Time',
            showlegend=False,
            xaxis=dict(gridcolor='lightgrey'),
            yaxis=dict(gridcolor='lightgrey'),
        )

        plot_save_path = os.path.join(save_dir, 'total_count_vs_batch_number.html')
        fig.write_html(plot_save_path)

    # RAM buffer for significant HDD speedup
    def preload_tif_files(self, image_folder, preload_buffer_percentage, ram_buffer, processed_images, preload_index):
        free_ram = psutil.virtual_memory().available
        preload_buffer_size = int(free_ram * preload_buffer_percentage / 100)
        image_size = 1 * 1024 * 1024 * 1024

        tiff_images = sorted([file for file in os.listdir(image_folder)
                             if file.endswith('.tif')],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))


        max_images = int(preload_buffer_size / image_size)

        # Keep track of the last loaded index separately
        last_loaded_index = preload_index

        while preload_index < len(tiff_images) and len(ram_buffer) < max_images:
            image_name = tiff_images[preload_index]
            if image_name not in processed_images and image_name not in ram_buffer.keys():
                image_path = os.path.join(image_folder, image_name)

                image_size = os.path.getsize(image_path) / (1024 * 1024)  # Convert to megabytes

                if len(ram_buffer) + 1 <= max_images:
                    with open(image_path, 'rb') as file:
                        image_data = file.read()
                        ram_buffer[preload_index]=(image_name, image_data)
                        last_loaded_index = preload_index  # Update last_loaded_index

            preload_index += 1

        return ram_buffer, preload_index, max_images, last_loaded_index

    # HDD specific function, takes out random read in favor of sequential read
    def predict_file_HDD(self, image_folder, save_dir, batch_size, 
                         preload_buffer_parameter, buffer_size, 
                         threshold,  debug=False, lower_bound=40, upper_bound=500):
        tiff_images = sorted([file for file in os.listdir(image_folder)
                              if file.endswith('.tif')],
                             key=lambda x: int(x.split('_')[1].split('.')[0]))

        image_dir = os.path.join(save_dir, 'images')

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        if not os.path.isdir(image_dir):
            os.makedirs(image_dir)

        csv_file = os.path.join(save_dir, 'cell_count.csv')

        image_index = 0  # Initialize image index
        totalCount = 0  # Initialize total count
        frame_queue = queue.Queue()
        futures = OrderedDict()
        ppcs = preProcessor_counter()  # Initialize the preprocessor object

        stdscr = curses.initscr()  # Initialize the curses screen
        curses.start_color()  # Start color manipulation

        
        # Initialize the list to store processed image names
        processed_images = []
        ram_buffer = OrderedDict()

        # Calculate the total number of batches based on the total number of images and the batch size
        total_batches = ceil(len(tiff_images) / batch_size)
        # Initialize the progress bar
        stdscr.move(0, 0)
        progress_bar = tqdm(total=total_batches, desc='Overall Progress', unit='batch', leave=True)
        progress_bar.set_postfix({'Total Count': totalCount})

        # Initialize lists to store batch numbers and total counts for plotting
        batch_times = []
        total_counts = []

        try:
            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                expected_image_index = 0  # Initialize the expected image index
                batch_number = 0  # Initialize the batch number
                preload_index = 0
                last_loaded_index = preload_index
               

                while expected_image_index < len(tiff_images):
                    # Refill buffer if required
                    if len(futures) < buffer_size and image_index < len(tiff_images):
                        # Preload tif files into RAM buffer
                        ram_buffer, preload_index, max_images, last_loaded_index = self.preload_tif_files(
                            image_folder, preload_buffer_parameter, ram_buffer, processed_images, last_loaded_index)
                        image_name, image_data = ram_buffer.pop(image_index)
                        stdscr.move(1, 0)  # move out of the progress bar's position
                        stdscr.clrtoeol()  # Clear the entire line
                        future = executor.submit(ppcs.process_image, None, 
                                                1, threshold=threshold,
                                                HDD = True, image_data=image_data,
                                                lower_bound=lower_bound, upper_bound=upper_bound)
                        futures[image_index] = (image_name, future)
                        image_index += 1

                    # Check if the expected future has completed
                    if expected_image_index in futures and futures[expected_image_index][1].done():
                        stdscr.move(0, 0)
                        stdscr.clrtoeol()  # Clear the entire line
                        stdscr.addstr(0, 0, str(progress_bar))
                        stdscr.refresh()
                        name, future = futures.pop(expected_image_index)

                        # Get the processed image
                        processed_image = future.result()

                        # Add the processed image to the queue
                        frame_queue.put((name, processed_image))

                        if debug == True:
                            frame_path = os.path.join(image_dir, name)
                            cv2.imwrite(frame_path, processed_image * 255)
                        
                        # Increment the preload_index to point to the next image to be fetched
                        preload_index += 1
                        expected_image_index += 1  # Move to the next expected image index
                        # Update processed images list
                        processed_images.append(name)

                        # Check if the batch is complete, or if the last image has been processed
                        if expected_image_index % batch_size == 0 or expected_image_index == len(tiff_images) - 1:
                            frame_queue.put(None)  # indicate end of batch
                            # process the queue
                            totalCount = self.process_queue(frame_queue,
                                                            totalCount,
                                                            csv_file)
                            batch_number += 1
                            # Update the progress bar
                            progress_bar.set_postfix({'Total Count': totalCount})
                            progress_bar.update(1)

                            # Update the lists for plotting
                            time_elapsed = batch_number * batch_size * 1240 / 432000
                            batch_times.append(time_elapsed)
                            total_counts.append(totalCount)
                            # Plot the line chart
                            self.plot_line_chart(batch_times, total_counts, save_dir)

                    else:
                        while len(ram_buffer) < max_images and preload_index < len(tiff_images):
                            ram_buffer, preload_index, max_images, last_loaded_index = self.preload_tif_files(image_folder, preload_buffer_parameter, ram_buffer, processed_images, preload_index)
                        time.sleep(0.1)  # Wait for the order to complete

                    # Move the progress bar to the top of the console
                    stdscr.move(0, 0)
                    stdscr.clrtoeol()  # Clear the entire line
                    stdscr.addstr(0, 0, str(progress_bar))
                    # Refresh the screen to update the progress bar's position
                    stdscr.refresh()

        finally:
            # Restore the terminal settings
            curses.nocbreak()
            stdscr.keypad(False)
            curses.echo()
            curses.curs_set(1)
            curses.endwin()
            os.system('cls' if os.name == 'nt' else 'clear')
            print("All images processed, total count: ", totalCount)