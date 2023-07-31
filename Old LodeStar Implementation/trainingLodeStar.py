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

matplotlib.use("TkAgg")
t, x, y, w = (1, 635, 178, 15)
training_image = dt.LoadImage(f"/mnt/h/TRAINING0620.tif")()._value / 256
crop = training_image[y:y+w, x:x+w]
plt.imshow(crop)
plt.axis("off")
plt.show()

model = dt.models.LodeSTAR(input_shape=(None, None, 3))

train_set = (
    dt.Value(crop)
    >> dt.Add(lambda: np.random.randn() * 0.1)
    >> dt.Gaussian(sigma=lambda: np.random.uniform(0, 0.2))
    >> dt.Multiply(lambda: np.random.uniform(0.6, 1.2))
)

model.fit(
    train_set,
    epochs=50,
    batch_size=8,
)

# model.save("lodeSTAR.h5py")

alpha = 0.2
cutoff = 0.99

#image = dt.LoadImage(f"/mnt/h/20230620/LTB4-500nM-2.5uL-1_200.tif")()._value / 256
#image = image[1, 0:200, 155:1300]
#image = rgb2gray(image)
#image = np.expand_dims(image, axis=-1)

tif_file = '/mnt/h/20230620/LTB4-500nM-2.5uL-1_*.tif' # whildcard to read all tif files
multi_tif = skimage.io.MultiImage(tif_file) # as many objects as tif files
length = len(multi_tif) # number of tif files
#length = 1


def process_image(i):
    csv_filename = f'/mnt/h/20230620/data/nodes20230620_{i}.csv'
    with open (csv_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['particle number', 'x', 'y', 'frame','picture'])
    images = multi_tif[i] # read each tif file
    fNumber = 0
    for frame in images:
        image = frame[0:200, 155:1300] # crop to channel only
        image = image / 256 # normalize !!
        detections = model.predict_and_detect(image[np.newaxis], 
                                              alpha=alpha, 
                                              beta=1-alpha, 
                                              cutoff=cutoff, 
                                              mode="quantile")[0]
        num_rows, num_cols = detections.shape

        if fNumber == 0:
            outputFig = plt.figure(figsize=(18,2.5))
            plt.imshow(image)
            plt.axis("off")
            plt.scatter(detections[:, 1], detections[:, 0], s=20, linewidths=1, marker="x", color="red")
            plt.savefig(f'/mnt/h/20230620/processed/LTB4-500nM-2.5uL-1_{i}.png')
            plt.close(outputFig)
            fNumber += 1

        with open(csv_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for j in range(num_rows):
                writer.writerow([j, detections[j, 1], detections[j, 0], fNumber, i])
            #del detections
            fNumber += 1

max_concurrent_tasks = 8
with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_tasks) as executor:
    executor.map(process_image, range(length))

#detections = autotracker.detect(pred[0], weights[0], beta=1-alpha, alpha=alpha, cutoff=cutoff, mode="constant")

#plt.imshow(image, cmap="gray")