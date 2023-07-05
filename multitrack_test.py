import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import deeptrack as dt

#tf.config.set_visible_devices([], 'GPU')
from deeptrack.extras import datasets
datasets.load('QuantumDots')

IMAGE_SIZE = 128

particle = dt.PointParticle(
    position=lambda: np.random.rand(2) * IMAGE_SIZE,
    z=lambda: np.random.randn() * 5,
    intensity=lambda: 1 + np.random.rand() * 9,
    position_unit="pixel",
)

number_of_particles = lambda: np.random.randint(10, 20)

particles = particle ^ number_of_particles

optics = dt.Fluorescence(
    NA=lambda: 0.6 + np.random.rand() * 0.2,
    wavelength=500e-9,
    resolution=1e-6,
    magnification=10,
    output_region=(0, 0, IMAGE_SIZE, IMAGE_SIZE),
)

normalization = dt.NormalizeMinMax(
    min=lambda: np.random.rand() * 0.4,
    max=lambda min: min + 0.1 + np.random.rand() * 0.5,
)

noise = dt.Poisson(
    snr=lambda: 4 + np.random.rand() * 3,
    background=normalization.min
)

imaged_particle = optics(particles)

dataset = imaged_particle >> normalization >> noise

CIRCLE_RADIUS = 3

X, Y = np.mgrid[:2*CIRCLE_RADIUS, :2*CIRCLE_RADIUS]

circle = (X - CIRCLE_RADIUS + 0.5)**2 + (Y - CIRCLE_RADIUS + 0.5)**2 < CIRCLE_RADIUS**2
circle = np.expand_dims(circle, axis=-1)

get_masks = dt.SampleToMasks(
    lambda: lambda image: circle,
    output_region=optics.output_region,
    merge_method="or"
)

def get_label(image_of_particles):
    return get_masks.update().resolve(image_of_particles)

NUMBER_OF_IMAGES = 4

#for _ in range(NUMBER_OF_IMAGES):
#    plt.figure(figsize=(15, 5))
#    dataset.update()
#    image_of_particle = dataset.resolve(skip_augmentations=True)
#    particle_label = get_label(image_of_particle)
#    plt.subplot(1, 2, 1)
#    plt.imshow(image_of_particle[..., 0], cmap="gray")
#    plt.subplot(1, 2, 2)
#    plt.imshow(particle_label[..., 0] * 1.0, cmap="gray")
#    plt.show()

import tensorflow.keras.backend as K
import tensorflow.keras.optimizers as optimizers

loss = dt.losses.flatten(
    dt.losses.weighted_crossentropy((10,1))
)
metric = dt.losses.flatten(
    dt.losses.weighted_crossentropy((1,1))
)
model = dt.models.UNet(
    (None, None, 1),
    conv_layers_dimensions=[16, 32, 64],
    base_conv_layers_dimensions=[128, 128],
    loss=loss,
    metrics=[metric],
    output_activation="sigmoid",
)
model.summary()

TRAIN_MODEL = True

validation_set_size = 100

validation_set = [dataset.update().resolve() for _ in range(validation_set_size)]
validation_labels = [get_label(image) for image in validation_set]

if TRAIN_MODEL:
    generator = dt.generators.ContinuousGenerator(
        dataset & (dataset >> get_label),
        batch_size = 16,
        min_data_size=1e3,
        max_data_size=1e4,
    )

    with generator:
        h = model.fit(generator,
            epochs=10,
            validation_data=(np.array(validation_set), np.array(validation_labels)),
        )
    
    model.compile(loss=metric, optimizer="adam")

    h2 = model.fit(generator, epochs=60, validation_data=(np.array(validation_set), np.array(validation_labels)))

else: 
    model_path = datasets.load_model("QuantumDots") 
    model.load_weights(model_path)

NUMBER_OF_IMAGES = 4


#for _ in range(NUMBER_OF_IMAGES):
#    plt.figure(figsize=(10, 10))
#    dataset.update()
#    image_of_particle = dataset.resolve(skip_augmentations=True)
    
#    predicted_mask = model.predict(np.array([image_of_particle]))
#    particle_label = get_label(image_of_particle)
#    plt.subplot(1, 3, 1)
#    plt.imshow(image_of_particle[..., 0], cmap="gray")
#    plt.subplot(1, 3, 2)
#    plt.imshow(particle_label[..., 0] * 1.0, cmap="gray")
#    plt.subplot(1, 3, 3)
#    plt.imshow(predicted_mask[0, ..., 0] > 0.5, cmap="gray")
#    plt.show()

import skimage.io
from skimage.color import rgb2gray

images = skimage.io.imread_collection("/mnt/h/20230621/LTB4-500nM-2.5uL-2_200.tif")
images = rgb2gray(images)
#images = np.expand_dims(images[:64], axis=-1)
#images = dt.NormalizeMinMax(0,1).resolve(list(images))

import IPython
to_predict_on = [images[i] / 3 + images[i + 1] / 3 + images[i-1] / 3 for i in range(1, len(images) - 1)]
predictions = model.predict(np.array(to_predict_on), batch_size=1)

for prediction, image in zip(predictions, images[1:-1]):
    plt.figure(figsize=(36, 5))
    mask = prediction[:,:,0] > 0.99
    
    cs = np.array(skimage.measure.regionprops(skimage.measure.label(mask)))
    
    cs = np.array([c["Centroid"] for c in cs])
    
    plt.subplot(1,2,1)
    plt.imshow(image[..., 0], vmax=0.1, cmap="gray")  
    plt.axis("off")
    plt.axis([0, 1400, 0, 200])
    
    plt.subplot(1,2,2)
    plt.imshow(image[..., 0], vmax=0.1, cmap="gray")
    plt.scatter(cs[:, 1], cs[:, 0], 100, facecolors="none", edgecolors="w")
    plt.axis("off")
    plt.axis([0, 1400, 0, 200])
    
    #IPython.display.clear_output(wait=True)
    plt.show()
    #plt.pause(0.1)
