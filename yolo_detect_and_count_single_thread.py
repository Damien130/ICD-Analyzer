import glob
import numpy as np
import random
import os
import cv2
import sort
import csv
import time
from PIL import Image, ImageSequence
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# Commented out IPython magic to ensure Python compatibility.
# %pip install ultralytics
import ultralytics
ultralytics.checks()
from ultralytics import YOLO

# Specify the frame interval
frame_interval = 1

brightness_factor = 4.5
contrast_iterations = 5

# Specify the directory path containing the TIFF files
tiff_directory = '/mnt/d/Benam Lab/20230620/'
background_path = '/mnt/d/Benam Lab/20230620/background/background.tif'

# Specify the output directory
output_directory = '/mnt/h/testing/png'

def process_image(frame, background_path):
    background = Image.open(background_path).convert('RGB')
    background = background.crop((125, 0, background.width - 125, background.height))

    # Crop the frame to remove the channel borders
    cropped_frame = frame.crop((125, 0, frame.width - 125, frame.height))
    background_cropped = background

    # Convert the frame to a NumPy array
    frame_array = cv2.cvtColor(np.array(cropped_frame), cv2.COLOR_RGB2BGR)
    background_array = cv2.cvtColor(np.array(background_cropped), cv2.COLOR_RGB2BGR)

    # Perform background subtraction
    subtracted_frame = cv2.absdiff(frame_array, background_array)

    # Enhance contrast of subtracted frame
    for i in range(contrast_iterations):
        subtracted_frame = enhance_contrast(subtracted_frame)

    # Increase brightness of subtracted frame
    subtracted_frame = increase_brightness(subtracted_frame, brightness_factor)

    # Denoise the subtracted frame
    subtracted_frame = denoise(subtracted_frame)

    return subtracted_frame

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

class YOLOv8_ObjectDetector:
    """
    A class for performing object detection on images and videos using YOLOv8.

    Args:
    ------------
        model_file (str): Path to the YOLOv8 model file or yolo model variant name in ths format: [variant].pt
        labels (list[str], optional): A list of class labels for the model. If None, uses the default labels from the model file.
        classes (list[int], optional): The classes we want to detect, use it to exclude certain classes from being detected,
            by default all classes in labels are detectable.
        conf (float, optional): Minimum confidence threshold for object detection.
        iou (float, optional): Minimum IOU threshold for non-max suppression.

    Attributes:
    --------------
        classes (list[str]): A list of class labels for the model ( a Dict is also acceptable).
        conf (float): Minimum confidence threshold for object detection.
        iou (float): Minimum IOU threshold for non-max suppression.
        model (YOLO): The YOLOv8 model used for object detection.
        model_name (str): The name of the YOLOv8 model file (without the .pt extension).

    Methods :
    -------------
        default_display: Returns a default display (ultralytics plot implementation) of the object detection results.
        custom_display: Returns a custom display of the object detection results.
        predict_video: Predicts objects in a video and saves the results to a file.
        predict_img: Predicts objects in an image and returns the detection results.

    """

    def __init__(self, model_file = 'yolov8n.pt', labels= None, classes = None, conf = 0.25, iou = 0.45 ):

        self.classes = classes
        self.conf = conf
        self.iou = iou

        self.model = YOLO(model_file)
        self.model_name = model_file.split('.')[0]
        self.results = None

        # if no labels are provided then use default COCO names 
        if labels == None:
            self.labels = self.model.names
        else:
            self.labels = labels

    def predict_img(self, img, verbose=True):
        """
        Runs object detection on a single image.

        Parameters
        ----------
            img (numpy.ndarray): Input image to perform object detection on.
            verbose (bool): Whether to print detection details.

        Returns:
        -----------
            'ultralytics.yolo.engine.results.Results': A YOLO results object that contains 
             details about detection results :
                    - Class IDs
                    - Bounding Boxes
                    - Confidence score
                    ...
        (pls refer to https://docs.ultralytics.com/reference/results/#results-api-reference for results API reference)

        """

        # Run the model on the input image with the given parameters
        results = self.model(img, classes=self.classes, conf=self.conf, iou=self.iou, verbose=verbose)

        # Save the original image and the results for further analysis if needed
        self.orig_img = img
        self.results = results[0]

        # Return the detection results
        return results[0]



    def default_display(self, show_conf=True, line_width=None, font_size=None, 
                        font='Arial.ttf', pil=False, example='abc'):
        """
        Displays the detected objects on the original input image.

        Parameters
        ----------
        show_conf : bool, optional
            Whether to show the confidence score of each detected object, by default True.
        line_width : int, optional
            The thickness of the bounding box line in pixels, by default None.
        font_size : int, optional
            The font size of the text label for each detected object, by default None.
        font : str, optional
            The font type of the text label for each detected object, by default 'Arial.ttf'.
        pil : bool, optional
            Whether to return a PIL Image object, by default False.
        example : str, optional
            A string to display on the example bounding box, by default 'abc'.

        Returns
        -------
        numpy.ndarray or PIL Image
            The original input image with the detected objects displayed as bounding boxes.
            If `pil=True`, a PIL Image object is returned instead.

        Raises
        ------
        ValueError
            If the input image has not been detected by calling the `predict_img()` method first.
        """
        # Check if the `predict_img()` method has been called before displaying the detected objects
        if self.results is None:
            raise ValueError('No detected objects to display. Call predict_img() method first.')
        
        # Call the plot() method of the `self.results` object to display the detected objects on the original image
        display_img = self.results.plot(show_conf, line_width, font_size, font, pil, example)
        
        # Return the displayed image
        return display_img

        

    def custom_display(self, colors, show_cls = True, show_conf = True):
        """
        Custom display method that draws bounding boxes and labels on the original image, 
        with additional options for showing class and confidence information.

        Parameters:
        -----------
        colors : list
            A list of tuples specifying the color of each class.
        show_cls : bool, optional
            Whether to show class information in the label text. Default is True.
        show_conf : bool, optional
            Whether to show confidence information in the label text. Default is True.

        Returns:
        --------
        numpy.ndarray
            The image with bounding boxes and labels drawn on it.
        """

        img = self.orig_img
        # calculate the bounding box thickness based on the image width and height
        bbx_thickness = (img.shape[0] + img.shape[1]) // 450

        for box in self.results.boxes:
            textString = ""

            # Extract object class and confidence score
            score = box.conf.item() * 100
            class_id = int(box.cls.item())

            x1 , y1 , x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)

            # Print detection info
            if show_cls:
                textString += f"{self.labels[class_id]}"

            if show_conf:
                textString += f" {score:,.2f}%"

            # Calculate font scale based on object size
            font = cv2.FONT_HERSHEY_COMPLEX
            fontScale = (((x2 - x1) / img.shape[0]) + ((y2 - y1) / img.shape[1])) / 2 * 2.5
            fontThickness = 1
            textSize, baseline = cv2.getTextSize(textString, font, fontScale, fontThickness)

            # Draw bounding box, a centroid and label on the image
            img = cv2.rectangle(img, (x1,y1), (x2,y2), colors[class_id], bbx_thickness)
            #center_coordinates = ((x1 + x2)//2, (y1 + y2) // 2)

            #img =  cv2.circle(img, center_coordinates, 5 , (0,0,255), -1)
            
             # If there are no details to show on the image
            if textString != "":
                if (y1 < textSize[1]):
                    y1 = y1 + textSize[1]
                else:
                    y1 -= 2
                # show the details text in a filled rectangle
                img = cv2.rectangle(img, (x1, y1), (x1 + textSize[0] , y1 -  textSize[1]), colors[class_id], cv2.FILLED)
                img = cv2.putText(img, textString , 
                    (x1, y1), font, 
                    fontScale,  (0, 0, 0), fontThickness, cv2.LINE_AA)

        return img


    def predict_video(self, video_path, save_dir, save_format="avi", display='custom', verbose=True, **display_args):
        """Runs object detection on each frame of a video and saves the output to a new video file.

        Args:
        ----------
            video_path (str): The path to the input video file.
            save_dir (str): The path to the directory where the output video file will be saved.
            save_format (str, optional): The format for the output video file. Defaults to "avi".
            display (str, optional): The type of display for the detection results. Defaults to 'custom'.
            verbose (bool, optional): Whether to print information about the video file and output file. Defaults to True.
            **display_args: Additional arguments to be passed to the display function.

        Returns:
        ------------
            None
        """
        # Open the input video file
        cap = cv2.VideoCapture(video_path)

        # Get the name of the input video file
        vid_name = os.path.basename(video_path)

        # Get the dimensions of each frame in the input video file
        width = int(cap.get(3))  # get `width`
        height = int(cap.get(4))  # get `height`

        # Create the directory for the output video file if it does not already exist
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        # Set the name and path for the output video file
        save_name = self.model_name + ' -- ' + vid_name.split('.')[0] + '.' + save_format
        save_file = os.path.join(save_dir, save_name)

        # Print information about the input and output video files if verbose is True
        if verbose:
            print("----------------------------")
            print(f"DETECTING OBJECTS IN : {vid_name} : ")
            print(f"RESOLUTION : {width}x{height}")
            print('SAVING TO :' + save_file)

        # Define an output VideoWriter object
        out = cv2.VideoWriter(save_file,
                              cv2.VideoWriter_fourcc(*"MJPG"),
                              60, (width, height))

        # Check if the input video file was opened correctly
        if not cap.isOpened():
            print("Error opening video stream or file")

        # Read each frame of the input video file
        while cap.isOpened():
            ret, frame = cap.read()

            # If the frame was not read successfully, break the loop
            if not ret:
                print("Error reading frame")
                break

            # Run object detection on the frame and calculate FPS
            beg = time.time()
            results = self.predict_img(frame, verbose=False)
            if results is None:
                print('***********************************************')
            fps = 1 / (time.time() - beg)

            # Display the detection results
            if display == 'default':
                frame = self.default_display(**display_args)
            elif display == 'custom':
                frame == self.custom_display(**display_args)

            # Display the FPS on the frame
            frame = cv2.putText(frame, f"FPS : {fps:,.2f}",
                                (5, 15), cv2.FONT_HERSHEY_COMPLEX,
                                0.5, (0, 0, 255), 1, cv2.LINE_AA)

            # Write the frame to the output video file
            out.write(frame)

            # Exit the loop if the 'q' button is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # After the loop release the cap and video writer
        cap.release()
        out.release()


class YOLOv8_ObjectCounter(YOLOv8_ObjectDetector):
    """
    A class for counting objects in images or videos using the YOLOv8 Object Detection model.

    Attributes:
    -----------
    model_file : str
        The filename of the YOLOv8 object detection model.
    labels : list or None
        The list of labels for the object detection model. If None, the labels will be loaded from the model file.
    classes : list or None
        The list of classes to detect. If None, all classes will be detected.
    conf : float
        The confidence threshold for object detection.
    iou : float
        The Intersection over Union (IoU) threshold for object detection.
    track_max_age : int
        The maximum age (in frames) of a track before it is deleted.
    track_min_hits : int
        The minimum number of hits required for a track to be considered valid.
    track_iou_threshold : float
        The IoU threshold for matching detections to existing tracks.

    Methods:
    --------
    predict_img(img, verbose=True)
        Predicts objects in a single image and counts them.
    predict_video(video_path, save_dir, save_format="avi", display='custom', verbose=True, **display_args)
        Predicts objects in a video and counts them.

    """

    def __init__(self, model_file = 'yolov8n.pt', labels= None, classes = None, conf = 0.25, iou = 0.45, 
                 track_max_age = 45, track_min_hits= 15, track_iou_threshold = 0.3 ):

        super().__init__(model_file , labels, classes, conf, iou)

        self.track_max_age = track_max_age
        self.track_min_hits = track_min_hits
        self.track_iou_threshold = track_iou_threshold


        

    def predict_video(self, image_folder, save_dir, save_format="avi",
                             display='custom', verbose=True, **display_args):

        tiff_files = sorted(glob.glob(os.path.join(image_folder, '*.tif')))

        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        for tiff_file in tqdm(tiff_files, desc='Processing frames'):
            # Get TIFF file name
            tiff_name = os.path.basename(tiff_file)

            # Read the multi-frame TIFF file
            image = Image.open(tiff_file)


            # Initialize object tracker
            tracker = sort.Sort(max_age=self.track_max_age, min_hits=self.track_min_hits,
                                iou_threshold=self.track_iou_threshold)

            # Initialize variables for object counting
            totalCount = []
            currentArray = np.empty((0, 5))
            csv_file = os.path.join(save_dir, 'object_count.csv')

            # Read the frames from the multi-frame TIFF file
            for page in tqdm(ImageSequence.Iterator(image), desc='Processing frames', leave=False):
                
                frame = process_image(page, background_path)
                frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                
                
                # Run object detection on the frame
                results = self.predict_img(frame, verbose=False)
                detections = np.empty((0, 5))
                for box in results.boxes:
                    score = box.conf.item() * 100
                    class_id = int(box.cls.item())

                    x1, y1, x2, y2 = np.squeeze(box.xyxy.cpu().numpy()).astype(int)

                    currentArray = np.array([x1, y1, x2, y2, score])
                    detections = np.vstack((detections, currentArray))

                # Update object tracker
                resultsTracker = tracker.update(detections)
                for result in resultsTracker:
                    x1, y1, x2, y2, id = result
                    x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2), int(id)

                    # If we haven't seen a particular object ID before, register it in a list
                    if totalCount.count(id) == 0:
                        totalCount.append(id)


                with open(csv_file, mode='a') as file:
                    writer = csv.writer(file)
                    writer.writerow([len(totalCount)])
                    print(len(totalCount))

                # the 'q' button is set as the
                # quitting button you may use any
                # desired button of your choice
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            print(len(totalCount))
            print(totalCount)