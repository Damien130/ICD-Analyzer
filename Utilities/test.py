import concurrent.futures
import queue

def process_frame(page, background_path):
    frame = process_image(page, background_path)
    frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
    return frame

def predict_video(self, image_folder, save_dir, save_format="avi",
                  display='custom', verbose=True, **display_args):

    tiff_files = sorted(glob.glob(os.path.join(image_folder, '*.tif')))

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    frame_queue = queue.Queue()  # Create a queue to store processed frames

    # Define a helper function to process frames and add them to the queue
    def process_and_enqueue(page):
        frame = process_frame(page, background_path)
        frame_queue.put(frame)

    # Set the number of frames to process before switching to detection
    frames_per_batch = 10
    frames_processed = 0

    # Process frames and perform object detection in parallel using multithreading
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for tiff_file in tqdm(tiff_files, desc='Processing frames'):
            # Read the multi-frame TIFF file
            image = Image.open(tiff_file)

            # Read the frames from the multi-frame TIFF file
            for page in ImageSequence.Iterator(image):
                # Process frames until the batch size is reached
                if frames_processed < frames_per_batch:
                    # Submit the processing task to the executor
                    future = executor.submit(process_and_enqueue, page)
                    futures.append(future)
                    frames_processed += 1
                else:
                    # Wait for all processing tasks to complete
                    concurrent.futures.wait(futures)

                    # Perform object detection on frames from the queue
                    while not frame_queue.empty():
                        frame = frame_queue.get()

                        # Run object detection on the frame and calculate FPS
                        results = self.predict_img(frame, verbose=False)
                        # Rest of your detection code...

                        # Display detection results
                        # Write detection results to a csv file
                        # Rest of your code...

                    # Reset the counter and clear the list of futures
                    frames_processed = 0
                    futures.clear()

    # Process any remaining frames in the queue
    while not frame_queue.empty():
        frame = frame_queue.get()

        # Run object detection on the frame and calculate FPS
        results = self.predict_img(frame, verbose=False)
        # Rest of your detection code...

        # Display detection results
        # Write detection results to a csv file
        # Rest of your code.....

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = []

    # Count the number of tiff cells
    total_cells = sum([len(list(ImageSequence.Iterator(Image.open(tiff_file)))) for tiff_file in tiff_files])

    # Set the number of cells per batch
    cells_per_batch = batch_size

    # Initialize the current cell count and batch list
    current_cell = 0
    batch = []

    for tiff_file in tqdm(tiff_files, desc='Processing frames'):
        # Read the multi-frame TIFF file
        image = Image.open(tiff_file)

        # Read the cells from the multi-frame TIFF file
        for page in ImageSequence.Iterator(image):
            batch.append(page)
            current_cell += 1

            if current_cell >= cells_per_batch:
                # Submit the batch for processing
                future = executor.submit(process_and_enqueue_batch, batch)
                futures.append(future)

                # Reset the batch and current cell count
                batch = []
                current_cell = 0

    # Submit the remaining cells as the last batch
    if batch:
        future = executor.submit(process_and_enqueue_batch, batch)
        futures.append(future)

    # Wait for all processing tasks to complete
    concurrent.futures.wait(futures)

    # Process the frames in the queue
    while not frame_queue.empty():
        frame = frame_queue.get()
        results = self.predict_img(frame, verbose=False)
        detections = np.empty((0, 5))
        # ...


# create a ThreadPoolExecutor object
        with ThreadPoolExecutor(max_workers=16) as executor:
            frame_counter = 0

            with tqdm(total=len(tiff_images), unit='image') as pbar:
                for image_index, image_name in enumerate(tiff_images):
                    image_path = os.path.join(image_folder, image_name)
                    frames = process_image(image_path, background_path)

                    for frame in frames:
                        frame_queue.put(frame)
                        frame_counter += 1

                    if (image_index + 1) % batch_size == 0:
                        frame_queue.put(None)
                        self.process_queue(frame_queue, tracker, totalCount, currentArray, csv_file)
                        frame_counter = 0
                    
                    pbar.set_postfix({'Total Count': len(totalCount)})
                    pbar.update(1)


with tqdm(total=len(tiff_images), desc='Preprocessing', unit='image') as pbar:
                
                for image_index, image_name in enumerate(tiff_images):
                    image_path = os.path.join(image_folder, image_name)
                    if len(futures) <= max_queue_length:
                        future = executor.submit(process_image, image_path, background_path)
                        futures.append((image_index, future))

                    for image_index, future in futures:
                        frames = future.result()
                        futures = [(i, f) for i, f in futures if f != future]  # Remove processed future
                        future = None  # Release reference to the processed future
                        for frame in frames:
                            frame_queue.put(frame)
                            frame_counter += 1

                        if (image_index + 1) % batch_size == 0:
                            frame_queue.put(None)
                            self.process_queue(frame_queue, tracker, totalCount, currentArray, csv_file)
                            frame_counter = 0
                        
                        pbar.set_postfix({'Total Count': len(totalCount),
                                        'Cache Size' : len(futures)})
                        #progress_bar.set_postfix({'queue size': queue.qsize()})
                        pbar.update(1)

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

        
    
    # release memory
    image.close()
    background.close()
    return frames