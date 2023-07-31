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
#import tensorflow as tf
import skimage.io
import csv

#def rgb2gray(rgb):
#    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

matplotlib.use("TkAgg")
t, x, y, w = (1, 670, 158, 10)
training_image = dt.LoadImage(f"/mnt/h/TRAINING0620.tif")()._value / 256
crop = training_image[y:y+w, x:x+w]
#crop = rgb2gray(crop)
#crop = np.expand_dims(crop, axis=-1)
#plt.imshow(crop, cmap="gray")
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

model.summary()

#model.save("lodeSTAR.h5py")

del tif_file
del multi_tif
tif_file = '/mnt/h/20230621/LTB4-500nM-2.5uL-2_*.tif' # whildcard to read all tif files
multi_tif = skimage.io.MultiImage(tif_file) # as many objects as tif files
length = len(multi_tif) # number of tif files

with open('/mnt/h/nodes20230621.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['particle number', 'x', 'y', 'frame','picture'])

for i in range(length):
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
        if fNumber == 0:
            outputFig = plt.figure(figsize=(18,2.5))
            plt.imshow(image)
            plt.axis("off")
            plt.scatter(detections[:, 1], detections[:, 0], color="r")
            #plt.show()
            plt.savefig(f'/mnt/h/20230621/processed/LTB4-500nM-2.5uL-1_{i}.png')
            plt.close(outputFig)
        #print(detections)
        num_rows, num_cols = detections.shape
        with open('/mnt/h/nodes20230621.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            for j in range(num_rows):
                writer.writerow([j, detections[j, 1], detections[j, 0], fNumber, i])
        del detections
        fNumber += 1