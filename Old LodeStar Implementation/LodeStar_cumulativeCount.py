import deeptrack as dt
import numpy as np
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
from PIL import Image
import skimage.io
import csv
import concurrent.futures
import os
import threading

os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '1'

# alpha is the probability of a pixel being a cell
alpha = 0.15
# cutoff is the threshold for a pixel to be considered a cell
cutoff = 0.999

# TkAgg (instead of Qt5Agg) is needed to display the plot in the GUI
matplotlib.use("TkAgg")

t, x, y, w = (1, 212, 151, 15)  # (frame, x, y, width)
training_image = dt.LoadImage(f"/mnt/d/Benam Lab/TRAINING0620_1.tif")()._value / 256
crop = training_image[y:y + w, x:x + w]  # crop the image
plt.imshow(crop)
plt.axis("off")
plt.show()

model = dt.models.LodeSTAR(input_shape=(None, None, 3))  # create the model

train_set = (
    dt.Value(crop)
    >> dt.Add(lambda: np.random.randn() * 0.1)
    >> dt.Gaussian(sigma=lambda: np.random.uniform(0, 0.2))
    >> dt.Multiply(lambda: np.random.uniform(0.6, 1.2))
)

# train the model
model.fit(
    train_set,
    epochs=30,
    batch_size=8,
)

length = 16
total_frames_per_picture = 1242

cell_counts = {}  # Shared dictionary to store cell counts for each frame
lock = threading.Lock()  # Lock object for synchronized access to the shared dictionary

# process_image is the function that will be run in parallel
def process_image(i):
    # create a csv file for each image
    csv_filename = f'/mnt/h/20230620/data/nodes20230620_{i}.csv'

    # read in the image
    image_file = f'/mnt/d/Benam Lab/20230620/LTB4-500nM-2.5uL-1_{i}.tif'
    images = skimage.io.imread(image_file)

    with lock:  # Acquire lock to access shared dictionary
        accumulated_frame_count = sum(len(cell_counts.get(j, [])) for j in range(i))  # Calculate accumulated frame count

    fNumber = accumulated_frame_count  # frame number
    frame_cell_counts = []  # number of cells in each frame
    passed_cells = set()  # set of cells that have passed through the image

    for frame_index, frame in enumerate(images):
        # crop the image
        image = frame[0:200, 155:1300]
        image = image / 256  # normalize the image
        # detect the cells
        # detections is a numpy array of shape (num_cells, 3)
        detections = model.predict_and_detect(image[np.newaxis],
                                              alpha=alpha,
                                              beta=1 - alpha,
                                              cutoff=cutoff,
                                              mode="quantile")[0]
        num_rows, num_cols = detections.shape

        # update the set of cells that have passed through the channel
        passed_cells.update(detections[:, 0])

        # write the data to the csv file
        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            with lock:  # Acquire lock to access shared dictionary
                for j in range(num_rows):
                    if detections[j, 0] in passed_cells:
                        frame_id = i * total_frames_per_picture + frame_index
                        writer.writerow([j, detections[j, 1],
                                         detections[j, 0], frame_id, i])
                        # Update the shared cell counts dictionary
                        cell_counts.setdefault(frame_id, set()).add(detections[j, 0])
            del detections  # delete the detections array to save memory

        # update the number of cells in each frame
        frame_cell_counts.append(len(passed_cells))

    with lock:  # Acquire lock to access shared dictionary
        # Update the shared cell counts dictionary
        cell_counts[i] = frame_cell_counts


# run the process_image function in parallel
max_concurrent_tasks = 16
with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
    executor.map(process_image, range(length))

# Calculate the cumulative cell counts
cumulative_counts = []
with lock:  # Acquire lock to access shared dictionary
    for frame_index in range(max(cell_counts.keys()) + 1):
        cell_set = cell_counts.get(frame_index, set())
        cumulative_counts.append(len(cell_set))

# plot the cumulative cell counts
plt.plot(cumulative_counts)
plt.xlabel("Frame")
plt.ylabel("Cumulative Cell Count")
plt.title("Cumulative Object Counting (Cells Passing Through Channel)")
plt.savefig('/mnt/h/20230620/data/cumulative_counts.png')

# write the cumulative cell counts to a csv file
with open('/mnt/h/20230620/data/cumulative_counts.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['frame', 'cumulative count'])
    for i, count in enumerate(cumulative_counts):
        writer.writerow([i, count])
