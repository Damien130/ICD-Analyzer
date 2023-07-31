import cv2
import os

def convert_png_to_mov(png_directory, output_file):
    # Get a list of all PNG files in the specified directory
    png_files = sorted([f for f in os.listdir(png_directory) if f.endswith('.png')], key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Get the dimensions of the first image
    first_image = cv2.imread(os.path.join(png_directory, png_files[0]))
    height, width, _ = first_image.shape

    # Create a VideoWriter object to save the MOV file
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or use 'XVID'
    fps = 30  # Frames per second
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    # Iterate through the PNG files and write each frame to the video
    for png_file in png_files:
        png_path = os.path.join(png_directory, png_file)
        frame = cv2.imread(png_path)
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()

# Example usage
png_directory = '/mnt/d/Benam/Data/hmm2'
output_file = '/mnt/i/hmm.mov'
convert_png_to_mov(png_directory, output_file)
