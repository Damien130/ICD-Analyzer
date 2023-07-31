import os
import curses
import time
        
def get_user_input(stdscr):

    curses.curs_set(1) # Show the cursor
    curses.echo() # Enable echoing of keys to the screen
    stdscr.addstr(14, 0, "Please enter the following parameters, enter (Q) to quit at any time:")

    while True:
        stdscr.addstr(16, 0, "Function: (C)omputer Vision, (B)inary Segmentation, or (D)emo: ")
        method = stdscr.getstr(16, 63, 1).decode()

        if method.lower() == 'q':
            stdscr.addstr(17, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
            stdscr.refresh()
            time.sleep(2)
            exit()
        elif method.lower() not in ['c', 'b', 'd']:
            stdscr.addstr(17, 0, "Invalid input. Please choose 'C', 'B', or 'D'.", curses.color_pair(2))
            stdscr.move(16, 63)
            stdscr.refresh()
            stdscr.getch()
            stdscr.clrtoeol()
        else:
            break

    if method.lower() == 'd':
        stdscr.addstr(17, 0, "You have entered demo mode", curses.color_pair(4))
        while True:
            stdscr.addstr(18, 0, "(C)omputer Vision or (B)inary Segmentation demo: ")
            demo_method = stdscr.getstr(18, 56, 1).decode()

            if demo_method.lower() == 'q':
                stdscr.addstr(19, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif demo_method.lower() not in ['c', 'b']:
                stdscr.addstr(19, 0, "Invalid input. Please choose 'C' or 'B'.", curses.color_pair(2))
                stdscr.move(18, 56)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                break
        
        while True:
            stdscr.addstr(19, 0, "Image folder: ")
            image_folder = stdscr.getstr(19, 14, curses.COLS - 14).decode()

            if image_folder.lower() == 'q':
                stdscr.addstr(20, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif not image_folder.strip():
                stdscr.addstr(20, 0, "Error: Please enter a valid Image folder path.", curses.color_pair(2))
                stdscr.move(19, 14)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                break

        if demo_method.lower() == 'c':
            while True:
                stdscr.addstr(20, 0, "Confidence threshold (default: 0.05): ")
                conf_thresh = stdscr.getstr(20, 40, 5).decode()
                if conf_thresh.lower() == 'q':
                    stdscr.addstr(21, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                    stdscr.refresh()
                    time.sleep(2)
                    exit()
                conf_thresh = float(conf_thresh) if conf_thresh else 0.05

                if conf_thresh > 1:
                    stdscr.addstr(21, 0, "Error: Confidence threshold should be between 0 and 1.", curses.color_pair(2))
                    stdscr.move(20, 40)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                elif conf_thresh < 0:
                    stdscr.addstr(21, 0, "Error: Confidence threshold should be between 0 and 1.", curses.color_pair(2))
                    stdscr.move(20, 40)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                elif conf_thresh == 0:
                    stdscr.addstr(21, 0, "Error: Confidence threshold cannot be 0.", curses.color_pair(2))
                    stdscr.move(20, 40)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                else:
                    break
        else:
            while True:
                stdscr.addstr(17, 0, "Threshold for binary segmentation (default: 0): ")
                threshold = stdscr.getstr(16, 50, 5).decode()
                if threshold.lower() == 'q':
                    stdscr.addstr(18, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                    stdscr.refresh()
                    time.sleep(2)
                    exit()
                threshold = float(threshold) if threshold else 0

                if threshold > 1:
                    stdscr.addstr(18, 0, "Error: Threshold should be between 0 and 1.",curses.color_pair(2))
                    stdscr.move(17, 50)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                elif threshold < 0:
                    stdscr.addstr(18, 0, "Error: Threshold should be between 0 and 1.",curses.color_pair(2))
                    stdscr.move(17, 50)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                else:
                    break

        
        while True:
            stdscr.addstr(18, 0, "Starting image number (default: 0): ")
            start_image_number = stdscr.getstr(18, 35, 5).decode()
            if start_image_number.lower() == 'q':
                stdscr.addstr(19, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            else:
                start_image_number = int(start_image_number) if start_image_number else 0
                break
        
        while True:
            stdscr.addstr(19, 0, "Generate images for the paper? (Y/N): ")
            generate_images = stdscr.getstr(19, 40, 1).decode()
            if generate_images.lower() == 'q':
                stdscr.addstr(20, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif generate_images.lower() not in ['y', 'n']:
                stdscr.addstr(20, 0, "Invalid input. Please choose 'Y' or 'N'.", curses.color_pair(2))
                stdscr.move(19, 40)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                generate_images = True if generate_images.lower() == 'y' else False
                break
        
        if generate_images:
            stdscr.addstr(20,0, "Output folder for generated images:")
            output_folder = stdscr.getstr(20, 40, curses.COLS - 40).decode()
            if output_folder.lower() == 'q':
                stdscr.addstr(21, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif not output_folder.strip():
                stdscr.addstr(21, 0, "Error: Please enter a valid output folder path.", curses.color_pair(2))
                stdscr.move(20, 40)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                output_folder = os.path.join(output_folder, 'images')
                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)
        else:
            output_folder = None
        
        return {
            'method': method,
            'demo_method': demo_method,
            'image_folder': image_folder,
            'start_image_number': start_image_number,
            'output_folder': output_folder,
            'generate_images': generate_images
        }
    else:
        while True:
            stdscr.addstr(17, 0, "Image folder: ")
            image_folder = stdscr.getstr(17, 14, curses.COLS - 14).decode()

            if image_folder.lower() == 'q':
                stdscr.addstr(18, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif not image_folder.strip():
                stdscr.addstr(18, 0, "Error: Please enter a valid Image folder path.", curses.color_pair(2))
                stdscr.move(17, 14)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                break

        while True:
            stdscr.addstr(18, 0, "Output folder: ")
            results_path = stdscr.getstr(18, 15, curses.COLS - 15).decode()

            if results_path.lower() == 'q':
                stdscr.addstr(19, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif not results_path.strip():
                stdscr.addstr(19, 0, "Error: Please enter a valid Output folder path.", curses.color_pair(2))
                stdscr.move(18, 15)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                break

        if method.lower() == 'c':
            while True:
                stdscr.addstr(19, 0, "Confidence threshold (default: 0.05): ")
                conf_thresh = stdscr.getstr(19, 40, 5).decode()
                if conf_thresh.lower() == 'q':
                    stdscr.addstr(20, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                    stdscr.refresh()
                    time.sleep(2)
                    exit()
                conf_thresh = float(conf_thresh) if conf_thresh else 0.05

                if conf_thresh > 1:
                    stdscr.addstr(20, 0, "Error: Confidence threshold should be between 0 and 1.", curses.color_pair(2))
                    stdscr.move(19, 40)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                elif conf_thresh < 0:
                    stdscr.addstr(20, 0, "Error: Confidence threshold should be between 0 and 1.", curses.color_pair(2))
                    stdscr.move(19, 40)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                elif conf_thresh == 0:
                    stdscr.addstr(20, 0, "Error: Confidence threshold cannot be 0.", curses.color_pair(2))
                    stdscr.move(19, 40)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                else:
                    break
        else:
            while True:
                stdscr.addstr(19, 0, "Threshold for binary segmentation (default: 0): ")
                threshold = stdscr.getstr(19, 50, 5).decode()
                if threshold.lower() == 'q':
                    stdscr.addstr(20, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                    stdscr.refresh()
                    time.sleep(2)
                    exit()
                threshold = float(threshold) if threshold else 0

                if threshold > 1:
                    stdscr.addstr(20, 0, "Error: Threshold should be between 0 and 1.",curses.color_pair(2))
                    stdscr.move(19, 50)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                elif threshold < 0:
                    stdscr.addstr(20, 0, "Error: Threshold should be between 0 and 1.",curses.color_pair(2))
                    stdscr.move(19, 50)
                    stdscr.refresh()
                    stdscr.getch()
                    stdscr.clrtoeol()
                else:
                    break

        while True:
            stdscr.addstr(20, 0, "Buffer size (default: 16): ")
            buffer_size = stdscr.getstr(20, 28, 5).decode()
            if buffer_size.lower() == 'q':
                stdscr.addstr(21, 0, "User initiated shutdown, exiting...",curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            buffer_size = int(buffer_size) if buffer_size else 16

            stdscr.addstr(21, 0, "Batch size (default: 1): ")
            if method.lower() == 'q':
                stdscr.addstr(22, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            batch_size = stdscr.getstr(21, 26, 5).decode()
            batch_size = int(batch_size) if batch_size else 1

            if batch_size > buffer_size:
                stdscr.addstr(22, 0, "Error: Batch size cannot be greater than buffer size.",curses.color_pair(2))
                stdscr.move(21, 26)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                break

        while True:
            stdscr.addstr(22, 0, "Debug mode (True/False, default: False): ")
            debug_param = stdscr.getstr(22, 42, 5).decode()
            debug = bool(debug_param.lower() in ['true', 't', 'yes', 'y', '1'])

            if debug_param.lower() == 'q':
                stdscr.addstr(23, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif debug_param.lower() not in ['true', 't', 'yes', 'y', '1', 'false', 'f', 'no', 'n', '0']:
                stdscr.addstr(23, 0, "Error: Please enter 'True' or 'False'.",curses.color_pair(2))
                stdscr.move(22, 42)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                break

        while True:
            stdscr.addstr(23, 0, "Is the source files on a mechanical Hard Drive? (True/False, default: False): ")
            hard_drive_param = stdscr.getstr(23, 77, 5).decode()
            hard_drive = bool(hard_drive_param.lower() in ['true', 't', 'yes', 'y', '1'])

            if hard_drive_param.lower() == 'q':
                stdscr.addstr(24, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            elif hard_drive_param.lower() not in ['true', 't', 'yes', 'y', '1', 'false', 'f', 'no', 'n', '0']:
                stdscr.addstr(24, 0, "Error: Please enter 'True' or 'False'.",curses.color_pair(2))
                stdscr.move(23, 87)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
            else:
                break

        if hard_drive == True:
            stdscr.addstr(24, 0, "How much RAM are you allocating? default 50%: ")
            ram_buffer = stdscr.getstr(24, 55, 5).decode()
            if ram_buffer.lower() == 'q':
                stdscr.addstr(25, 0, "User initiated shutdown, exiting...", curses.color_pair(2))
                stdscr.refresh()
                time.sleep(2)
                exit()
            ram_buffer = float(ram_buffer) if ram_buffer else 50
            if ram_buffer > 100:
                stdscr.addstr(25, 0, "Error: RAM buffer cannot be greater than 100%.",curses.color_pair(2))
                stdscr.move(24, 65)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
                ram_buffer = 50
            elif ram_buffer < 0:
                stdscr.addstr(25, 0, "Error: RAM buffer cannot be less than 0%.",curses.color_pair(2))
                stdscr.move(24, 65)
                stdscr.refresh()
                stdscr.getch()
                stdscr.clrtoeol()
                ram_buffer = 50
        else:
            ram_buffer = 0


        # Use locals() to retrieve the local variables within the function
        local_vars = locals()

        # Filter the local variables for conf_thresh and threshold
        params = {key: value for key, value in local_vars.items() if key in ['conf_thresh', 'threshold']}

        # Add other parameters to the dictionary
        params.update({
            'method': method,
            'image_folder': image_folder,
            'results_path': results_path,
            'batch_size': batch_size,
            'buffer_size': buffer_size,
            'debug': debug,
            'hard_drive': hard_drive,
            'ram_buffer': ram_buffer
        })

        return params
    

# Function to execute the Object Detection ('OD') method
def run_object_detection_ssd(args):
    from preProcessing import preProcessor_counter, preProcessor_CV
    from SCANNERsrc import segmentation_ObjectCounter as segCounter

    counter = segCounter()
    os.system('cls' if os.name == 'nt' else 'clear')
    # start process
    counter.predict_file(image_folder=args['image_folder'], 
                         save_dir=args['results_path'], 
                         batch_size=args['batch_size'], 
                         buffer_size=args['buffer_size'], 
                         threshold=args['threshold'], 
                         debug=args['debug'])
    
def run_object_detection_hdd(args):
    from SCANNERsrc import segmentation_ObjectCounter as segCounter

    counter = segCounter()
    os.system('cls' if os.name == 'nt' else 'clear')
    # start process
    counter.predict_file_HDD(image_folder=args['image_folder'],
                                save_dir=args['results_path'],
                                batch_size=args['batch_size'],
                                preload_buffer_parameter=args['ram_buffer'],
                                buffer_size=args['buffer_size'],
                                threshold=args['threshold'],
                                debug=args['debug'])

# Function to execute the Computer Vision ('CV') method
def run_computer_vision(args):
    from ultralytics import YOLO
    from yolo_detect_and_count import YOLOv8_ObjectDetector, YOLOv8_ObjectCounter

    # load model
    yolo_names = ['best.pt']

    counters = []
    for yolo_name in yolo_names:
        counter = YOLOv8_ObjectCounter(yolo_name, conf=args['conf_thresh'])
        counters.append(counter)
    os.system('cls' if os.name == 'nt' else 'clear')
    for counter in counters:
        counter.predict_video(image_folder=args['image_folder'], 
                              save_dir=args['results_path'], 
                              batch_size=args['batch_size'])

def run_object_detection_demo(args):
    from SCANNERdemo import binarySegmentationDemo

    # start process
    demo = binarySegmentationDemo(folder_path=args['image_folder'],
                                    start_image_number=args['start_image_number'],
                                    output_path=args['output_folder'],
                                    gen_images=args['generate_images'])
    os.system('cls' if os.name == 'nt' else 'clear')
    demo.process_images_and_show_video()
        
def main_menu(stdscr):
    stdscr = curses.initscr() # Initialize the curses screen
    curses.start_color() # Start color manipulation
    # Set the terminal to accept special key inputs (e.g., arrow keys)
    stdscr.keypad(True)
    # Disable automatic echoing of keys to the screen
    curses.noecho()
    # Hide the cursor
    curses.curs_set(0)
    # Initialize color pairs
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)  # White on Black
    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)    # Red on Black
    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Green on Black
    curses.init_pair(4, curses.COLOR_MAGENTA, curses.COLOR_BLACK) # Magenta on Black

    # ASCII art for SCANNER
    scanner_ascii_art = """
Scalable Cell Analysis using Neural Networks for Enhanced Recognition
    ##############################################################
    ##############################################################
    ##░██████╗░█████╗░░█████╗░███╗░░██╗███╗░░██╗███████╗██████╗░##
    ##██╔════╝██╔══██╗██╔══██╗████╗░██║████╗░██║██╔════╝██╔══██╗##
    ##╚█████╗░██║░░╚═╝███████║██╔██╗██║██╔██╗██║█████╗░░██████╔╝##
    ##░╚═══██╗██║░░██╗██╔══██║██║╚████║██║╚████║██╔══╝░░██╔══██╗##
    ##██████╔╝╚█████╔╝██║░░██║██║░╚███║██║░╚███║███████╗██║░░██║##
    ##╚═════╝░░╚════╝░╚═╝░░╚═╝╚═╝░░╚══╝╚═╝░░╚══╝╚══════╝╚═╝░░╚═╝##
    ##############################################################
    ##############################################################"""

    # Print the header lines
    stdscr.addstr(0, 0, "Benam Lab SCANNER 1.1", curses.color_pair(1))
    stdscr.addstr(1, 0, scanner_ascii_art, curses.color_pair(3))
    stdscr.refresh()
    time.sleep(1)

    # user input
    user_input = get_user_input(stdscr)

    

    # call the appropriate function based on the user input
    if user_input['method'].lower() == 'c':
        # check if image folder exists, if not create it
        if not os.path.isdir(user_input['results_path']):
            os.makedirs(user_input['results_path'])
        run_computer_vision(user_input)
    elif user_input['method'].lower() == 'b':
        if not os.path.isdir(user_input['results_path']):
            os.makedirs(user_input['results_path'])
        if user_input['hard_drive']:
            run_object_detection_hdd(user_input)
        else:
            run_object_detection_ssd(user_input)
    elif user_input['method'].lower() == 'd':
        if user_input['demo_method'].lower() == 'b':
            run_object_detection_demo(user_input)
        else:
            pass
    else:
        stdscr.addstr(28, 0, "Invalid method. Please choose 'c', 'b' or 'd'.", curses.color_pair(2))
        stdscr.refresh()
        time.sleep(2)
        main_menu(stdscr)

if __name__ == '__main__':
    curses.wrapper(main_menu)