import deeptrack as dt
import numpy as np
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import *
#import tensorflow as tf
import skimage.io
from skimage.color import rgb2gray

#def rgb2gray(rgb):
#    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

matplotlib.use("TkAgg")
t, x, y, w = (1, 985, 150, 20)
training_image = dt.LoadImage(f"/mnt/h/20230620/LTB4-500nM-2.5uL-1_39.tif")()._value / 256
crop = training_image[1, y:y+w, x:x+w]
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
    epochs=90,
    batch_size=8,
)

alpha = 0.17
cutoff = 0.99

#image = dt.LoadImage(f"/mnt/h/20230620/LTB4-500nM-2.5uL-1_200.tif")()._value / 256
#image = image[1, 0:200, 155:1300]
#image = rgb2gray(image)
#image = np.expand_dims(image, axis=-1)
#images = skimage.io.imread_collection("/mnt/h/20230620/LTB4-500nM-2.5uL-1_200.tif")
#images = skimage.util.crop(images,((0,0),(130,130),(0,0),(0,0)))


tif_file = '/mnt/h/20230620/LTB4-500nM-2.5uL-1_*.tif' # whildcard to read all tif files
multi_tif = skimage.io.MultiImage(tif_file) # as many objects as tif files
#length = len(multi_tif) # number of tif files
length = 2

for i in range(length):
    images = multi_tif[i] # read each tif file
    for frame in images:
        image = frame[0:200, 155:1300] # crop to channel only
        detections = model.predict_and_detect(image[np.newaxis], 
                                              alpha=alpha, 
                                              beta=1-alpha, 
                                              cutoff=cutoff, 
                                              mode="quantile")[0]
        plt.figure(figsize=(18,2.5))
        plt.imshow(image)
        plt.axis("off")
        plt.scatter(detections[:, 1], detections[:, 0], color="r")
        plt.show()



#detections = autotracker.detect(pred[0], weights[0], beta=1-alpha, alpha=alpha, cutoff=cutoff, mode="constant")

#plt.imshow(image, cmap="gray")
