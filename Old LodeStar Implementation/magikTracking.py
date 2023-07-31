import deeptrack as dt
from deeptrack.models.gnns.generators import GraphGenerator

import tensorflow as tf

import pandas as pd
import numpy as np

from deeptrack.extras import datasets

import logging
logging.disable(logging.WARNING)

datasets.load("BFC2Cells")

nodesdf = pd.read_csv("datasets/BFC2DLMuSCTra/nodesdf.csv")

# normalize centroids between 0 and 1
nodesdf.loc[:, nodesdf.columns.str.contains("centroid")] = (
    nodesdf.loc[:, nodesdf.columns.str.contains("centroid")]
    / np.array([1000.0, 1000.0])
)

# display the first 20 rows of the dataframe
nodesdf.head(20)

parenthood = pd.read_csv("datasets/BFC2DLMuSCTra/parenthood.csv")

# display the first 10 rows of the dataframe
parenthood.head(10)

# Output type
_OUTPUT_TYPE = "edges"

radius = 0.2

variables = dt.DummyFeature(
    radius=radius,
    output_type=_OUTPUT_TYPE,
    nofframes=3, # time window to associate nodes (in frames) 
)

model = dt.models.gnns.MAGIK(
    dense_layer_dimensions=(64, 96,),      # number of features in each dense encoder layer
    base_layer_dimensions=(96, 96, 96),    # Latent dimension throughout the message passing layers
    number_of_node_features=2,             # Number of node features in the graphs
    number_of_edge_features=1,             # Number of edge features in the graphs
    number_of_edge_outputs=1,              # Number of predicted features
    edge_output_activation="sigmoid",      # Activation function for the output layer
    output_type=_OUTPUT_TYPE,              # Output type. Either "edges", "nodes", or "graph"
)

# Compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = 'binary_crossentropy',
    metrics=['accuracy'],
)

model.summary()

_LOAD_MODEL = True

if _LOAD_MODEL:
    print("Loading model...")
    model.load_weights("datasets/BFC2DLMuSCTra/MAGIK.h5")
else:
    generator = GraphGenerator(
        nodesdf=nodesdf,
        properties=["centroid"],
        parenthood=parenthood,
        min_data_size=511,
        max_data_size=512,
        batch_size=8,
        **variables.properties()
    )
    
    with generator:
        model.fit(generator, epochs=10)

