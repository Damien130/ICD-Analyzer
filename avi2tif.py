import cv2
import os
import imageio

def convert_avi_to_tif(input_video_path, output_folder, frame_interval=1000):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the number of TIFF files needed
    num_tif_files = total_frames // frame_interval
    if total_frames % frame_interval != 0:
        num_tif_files += 1
    i = 1
    for tif_file_num in range(num_tif_files):
        # Create a new TIFF file
        tif_file_path = os.path.join(output_folder, f'output_{tif_file_num + 1}.tif')

        # Create an imageio writer without specifying fps
        writer = imageio.get_writer(tif_file_path)

        # Write frames to the TIFF file
        for frame_num in range(frame_interval):
            ret, frame = cap.read()
            if not ret:
                break
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        writer.close()

    cap.release()

# Example usage
if __name__ == "__main__":
    input_video_path = '/mnt/d/Benam/ICDExample/Media1.avi'
    output_folder = '/mnt/d/Benam/ICDExample/original'
    frame_interval = 500

convert_avi_to_tif(input_video_path, output_folder, frame_interval)
