import dearpygui.dearpygui as dpg
from preProcessing import preProcessor_counter, preProcessor_CV
from SCANNERsrc import segmentation_ObjectCounter as segCounter

counter = segCounter()
# Global variables
current_callback = None  # Global variable to track the current callback
input_directory = ""
output_directory = ""
lower_gate_size = 0
upper_gate_size = 0
window_width = 1280
window_height = 720
progress = 0
cell_count = 0
totalCount = 0

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

def start_scan(sender, app_data):
    print('Scan was started.')
    print("Input Directory: ", input_directory)
    print("Output Directory: ", output_directory)
    print("Upper Gate Size: ", upper_gate_size)
    print("Lower Gate Size: ", lower_gate_size)
    counter.predict_video(video_folder=input_directory,
                          save_dir = output_directory,
                          lower_bound = lower_gate_size,
                          upper_bound = upper_gate_size,
                          totalCount=totalCount)
    
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
        # Add a button for starting the scan
        dpg.add_button(label="Start Scan", callback=start_scan)
        # Draw a vertical line to separate input and output windows
        dpg.draw_line((200, 0), (200, 720), color=(255, 255, 255, 255), thickness=1.0)

    #with dpg.window(label="Tutorial", width=500, height=500):
    #    dpg.add_simple_plot(label="Simpleplot1", default_value=(0.3, 0.9, 0.5, 0.3), height=300)
    #    dpg.add_simple_plot(label="Simpleplot2", default_value=(0.3, 0.9, 2.5, 8.9), overlay="Overlaying", height=180,
    #                        histogram=True)
    
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