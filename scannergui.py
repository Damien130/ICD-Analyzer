import dearpygui.dearpygui as dpg
from preProcessing import preProcessor_counter
import cv2
import os
import numpy as np
import csv
import queue
from skimage import  measure
from math import ceil  # Import the ceil function
import dearpygui.dearpygui as dpg

# Global variables
current_callback = None  # Global variable to track the current callback
input_directory = ""
output_directory = ""
lower_gate_size = 0
upper_gate_size = 0
window_width = 1280
window_height = 720
progress = 0
cell_count = [0]
flow_rate = [0]
time_elapsed = [0]
FPS = 30

dpg.create_context()
dpg.create_viewport(title='SCANNER', width=1280, height=720)
dpg.setup_dearpygui()

def input_callback(sender, app_data):
    global input_directory
    print('OK was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)
    input_directory = app_data["file_path_name"]
    print(input_directory)

def output_callback(sender, app_data):
    global output_directory
    print('OK was clicked.')
    print("Sender: ", sender)
    print("App Data: ", app_data)
    output_directory = app_data["file_path_name"]
    print(output_directory)

def file_dialog_callback(sender, app_data):
    global current_callback
    if current_callback:
        current_callback(sender, app_data)

def cancel_callback(sender, app_data):
    print('Cancel was clicked.')

dpg.add_file_dialog(
    directory_selector=True, show=False, callback=file_dialog_callback, tag="file_dialog_id",
    cancel_callback=cancel_callback, width=700 ,height=400)

def setUpperGateSize(sender, app_data):
    global upper_gate_size
    print('Upper Gate Size was set.')
    upper_gate_size = int(dpg.get_value("upper_gate_size_input"))
    print(upper_gate_size)

def setLowerGateSize(sender, app_data):
    global lower_gate_size
    print('Lower Gate Size was set.')
    lower_gate_size = int(dpg.get_value("lower_gate_size_input"))
    print(lower_gate_size)

def set_current_callback(callback):
    global current_callback
    current_callback = callback
    dpg.show_item("file_dialog_id")

def setFPS(sender, app_data):
    global FPS
    print('FPS was set.')
    FPS = int(dpg.get_value("FPS_input"))
    print(FPS)

def start_scan(sender, app_data):
    print('Scan was started.')
    print("Input Directory: ", input_directory)
    print("Output Directory: ", output_directory)
    print("Upper Gate Size: ", upper_gate_size)
    print("Lower Gate Size: ", lower_gate_size)
    predict_video(video_folder=input_directory,
                        save_dir = output_directory,
                        lower_bound = lower_gate_size,
                        upper_bound = upper_gate_size)
                        #totalCount=cell_count,
                        #objectCount=flow_rate)
    
def update_plots(Time, YTotal, YRate):
    dpg.set_value('lePlot', [Time, YTotal])
    dpg.set_axis_limits('x_axis_lePlot', 0, Time[-1]+5)
    dpg.set_axis_limits('y_axis_lePlot', 0, YTotal[-1]+5)

    dpg.set_value('FlowPlot', [Time, YRate])
    dpg.set_axis_limits('x_axis_FlowPlot', 0, Time[-1]+5)
    dpg.set_axis_limits('y_axis_FlowPlot', 0, max(YRate)+5)

def predict_video(video_folder, save_dir, 
                    batch_size = 1, buffer_size = 16, threshold = 0.05, debug=False,
                    lower_bound = 40, upper_bound = 500, totalCount = [0],
                    objectCount = [0]):
    video_stream = sorted([file for file in os.listdir(video_folder) 
                            if file.endswith('.mp4')], 
                            key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    image_dir = os.path.join(save_dir, 'images')
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    if not os.path.isdir(image_dir):
        os.makedirs(image_dir)
    
    csv_file = os.path.join(save_dir, 'cell_count.csv')

    image_index = 0 # Initialize image index
    frame_queue = queue.Queue()
    ppcs = preProcessor_counter() # Initialize the preprocessor object

    # Calculate the total number of batches based on the total number of images and the batch size
    total_batches = ceil(len(video_stream) / batch_size)
    print ("acquired stream: ", video_stream)
    image_name = video_stream[image_index]
    image_path = os.path.join(video_folder, image_name)
    cap = cv2.VideoCapture(image_path)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Frame count: ", frame_count)
    chunk_size = 100
    num_chunks = ceil(frame_count / chunk_size)
    
    # Initialize lists to store batch numbers and total counts for plotting
    batch_times = []
    total_counts = []

    expected_image_index = 0  # Initialize the expected image index
    while expected_image_index < len(video_stream):

        for chunk_index in range(num_chunks):
            start_frame = chunk_index * chunk_size
            end_frame = min((chunk_index + 1) * chunk_size, frame_count)
            print("start: ", start_frame)
            print("finish: ", end_frame)
            future = ppcs.process_video(cap, start_frame, end_frame, 1, threshold=threshold,
                                    lower_bound=lower_bound, upper_bound=upper_bound)
            # Get the processed image
            processed_image = future

            if debug:
                frame_path = os.path.join(video_folder, image_name)
                cv2.imwrite(frame_path, processed_image * 255)


            # Add the processed image to the queue
            frame_queue.put((image_name, processed_image))
            frame_queue.put(None)  # indicate end of batch

            # process the queue
            objectCount.append(process_queue_gui(frame_queue, csv_file))
            chunk_index += 1

            # Calculate the sum of all values in the objectCount list and append it to the totalCount list
            totalCount.append(sum(objectCount))
            time_elapsed.append(end_frame / FPS)
            print("Time elapsed: ", time_elapsed)
            update_plots(time_elapsed, totalCount, objectCount)
            

        # Check if the batch is complete, or if the last image has been processed
        #if expected_image_index % batch_size == 0 or expected_image_index == len(tiff_images) - 1:
        
        # process the queue
        #totalCount = process_queue(frame_queue, totalCount, csv_file)

def process_queue_gui(queue, csv_file):
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

            # write to csv file
            with open(csv_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([image_name, object_count])
            
            processed_images += 1
        return object_count
    
def create_main_window(sender, app_data):
    # Close splash window
    dpg.delete_item("splash")
    # Create main window for scanner
    with dpg.window(label="SCANNER", tag='SCANNER', no_close=True, no_move=True, 
                    no_resize=True, no_title_bar=True, no_background=True, 
                    width=1280, height=720) as scanner:
        # Add a button for selecting the input video
        dpg.add_button(label="Select Input", callback=lambda: set_current_callback(input_callback))
        # Add a button for selecting the output directory
        dpg.add_button(label="Select Output Directory", callback=lambda: set_current_callback(output_callback))

        # Create a group Upper gate
        with dpg.group(horizontal=True):
            # Add text explaining "upper boundary"
            dpg.add_text("Upper Gate Size:")
            # Add user input box
            dpg.add_input_text(label="", width=30, tag="upper_gate_size_input")
            dpg.add_button(label="Set", callback = setUpperGateSize)
        
        with dpg.group(horizontal=True):
            # Add text explaining "lower boundary"
            dpg.add_text("Lower Gate Size:")
            # Add user input box
            dpg.add_input_text(label="", width=30, tag="lower_gate_size_input")
            dpg.add_button(label="Set", callback = setLowerGateSize)

        with dpg.group(horizontal=True):
            # Add text explaining "lower boundary"
            dpg.add_text("Microscopy FPS")
            # Add user input box
            dpg.add_input_text(label="", width=30, tag="FPS_input")
            dpg.add_button(label="Set", callback = setFPS)
        # Add a button for starting the scan
        dpg.add_button(label="Start Scan", callback=start_scan)
        # Draw a vertical line to separate input and output windows
        dpg.draw_line((200, 0), (200, 720), color=(255, 255, 255, 255), thickness=1.0)

    with dpg.window(label="Statistics", width=1050, height=700, pos=(220,10)):
        with dpg.plot(label="Total Count", width=1030, height=330,  tag="TotalCountPlot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time Elapsed (Seconds)", tag="x_axis_lePlot")
            dpg.add_plot_axis(dpg.mvYAxis, label="Total Cell Count", tag="y_axis_lePlot")
            dpg.set_axis_limits('x_axis_lePlot', 0, 10)
            dpg.set_axis_limits('y_axis_lePlot', 0, 10)
            dpg.add_line_series(time_elapsed, cell_count, label="Total", parent="x_axis_lePlot", tag="lePlot")

        with dpg.plot(label="Cell Flow Rate", width=1030, height=330,  tag="FlowRatePlot"):
            dpg.add_plot_axis(dpg.mvXAxis, label="Time Elapsed (Seconds)", tag="x_axis_FlowPlot")
            dpg.add_plot_axis(dpg.mvYAxis, label="Instataneous Flow Rate", tag="y_axis_FlowPlot")
            dpg.set_axis_limits('x_axis_FlowPlot', 0, 10)
            dpg.set_axis_limits('y_axis_FlowPlot', 0, 10)
            dpg.add_line_series(time_elapsed, cell_count, label="Flow", parent="x_axis_FlowPlot", tag="FlowPlot")    
    
    dpg.set_primary_window("SCANNER", True)



# load GUI textures
SPLw, SPLh, SPLc, SPL = dpg.load_image("GUI/SCANNER.png")

with dpg.texture_registry(show=False):
    dpg.add_static_texture(width=SPLw, height=SPLh, default_value=SPL, tag="splashpic")

# Create splash window for scanner
with dpg.window(label="Splashscreen", no_title_bar=True, no_background=True, 
                width=SPLw+50, height=SPLh+50, pos=[window_width/2 - SPLw/2,
                                             window_height/2 - SPLh/2],
                                             tag="splash"):
    splashbutton = dpg.add_image_button(texture_tag="splashpic" , label="Welcome to SCANNER", 
                         width=SPLw, height=SPLh, callback=create_main_window)
    splashtext = dpg.add_text(" Welcome to SCANNER, Click to continue")



dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()