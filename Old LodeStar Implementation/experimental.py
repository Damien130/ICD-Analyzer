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
passed_cells = set()

def process_image(i):
    # create a csv file for each image
    csv_filename = f'/mnt/h/20230620/data/nodes20230620_{i}.csv'

    # read in the image
    image_file = f'/mnt/d/Benam Lab/20230620/LTB4-500nM-2.5uL-1_{i}.tif'
    images = skimage.io.imread(image_file)

    # crop the image
    image = frame[0:200, 155:1300]
    image = image / 256  # normalize the image
    
    for frame in image:
        detections = model.predict_and_detect(image[np.newaxis],
                                              alpha=alpha,
                                              beta=1 - alpha,
                                              cutoff=cutoff,
                                              mode="quantile")[0]
        

# run the process_image function in parallel
max_concurrent_tasks = 16
with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
    executor.map(process_image, range(length))