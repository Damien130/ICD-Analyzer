import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import csv
import imageio
import os

tif_file = '/mnt/d/Benam Lab/20230620/LTB4-500nM-2.5uL-1_0.tif'
multi_tif = skimage.io.imread_collection(tif_file)
length = len(multi_tif)

# Define the desired figure width and panel density
fig_width = 10
panel_density = 80
fps = 68

png_folder = f'/mnt/h/20230620/processed/png_frames'
os.makedirs(png_folder, exist_ok=True)  # Create folder for PNG frames

for i in range(length):
    image_frames = multi_tif[i]
    frame_index = 0

    for frame in image_frames:
        image = frame[0:200, 155:1300] / 256

        csv_filename = f'/mnt/h/20230620/data/nodes20230620_{i}.csv'
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            detections = []
            for row in reader:
                frame_idx = int(row[3])  # Column 3 for frame index
                if frame_index == frame_idx:
                    x = float(row[1])  # Column 1 for x-coordinate
                    y = float(row[2])  # Column 2 for y-coordinate
                    detections.append([x, y])
            detections = np.array(detections)

        # Calculate the figure height based on the aspect ratio and panel density
        image_height, image_width = image.shape[:2]
        aspect_ratio = image_width / image_height
        fig_height = fig_width / aspect_ratio
        fig_height_pixels = fig_height * panel_density

        fig, ax = plt.subplots(figsize=(fig_width, fig_height_pixels / panel_density))
        ax.imshow(image)
        ax.axis("off")
        ax.scatter(detections[:, 0], detections[:, 1], s=35,
                   linewidths=0.5, marker="o", facecolors='none', edgecolors='r')

        # Save the frame as a PNG file
        output_frame_filename = f'{png_folder}/LTB4-500nM-2.5uL-1_{i}_frame{frame_index}.png'
        fig.savefig(output_frame_filename)
        plt.close(fig)
        frame_index += 1

# Create a list of all PNG files in the folder
png_files = sorted([f for f in os.listdir(png_folder) if f.endswith('.png')])

# Read each PNG file and add it to a list of frames
frames = []
for png_file in png_files:
    image = imageio.imread(os.path.join(png_folder, png_file))
    frames.append(image)

# Define the output GIF file path
output_gif_file = '/mnt/h/20230620/processed/animation.gif'

# Save the frames as a GIF file
imageio.mimsave(output_gif_file, frames, format='GIF')
