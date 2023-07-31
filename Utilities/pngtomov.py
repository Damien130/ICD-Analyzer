import cv2
import glob
import os

# Specify the directory path containing the PNG files
png_directory = '/mnt/d/Benam/Data/hmm2'

# Specify the output video file path
output_video_file = '/mnt/j/hmm.mov'

# Get the list of PNG files in the directory
png_files = glob.glob(os.path.join(png_directory, '*.png'))

# Sort the PNG files based on the frame index
#png_files.sort(key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2].split('.')[0])))

# Open the first PNG file to get the frame size
first_frame = cv2.imread(png_files[0])
frame_size = (first_frame.shape[1], first_frame.shape[0])

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc('F','F','V','1')
video_writer = cv2.VideoWriter(output_video_file, fourcc, 30.0, frame_size, isColor=True)

# Iterate over the sorted PNG files and write each frame to the video
for png_file in png_files:
    frame = cv2.imread(png_file)
    video_writer.write(frame)

# Release the VideoWriter and close the video file
video_writer.release()

print(f"Video saved: {output_video_file}")