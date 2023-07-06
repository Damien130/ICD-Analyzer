import skimage.io
import matplotlib.pyplot as plt
import numpy as np
import csv

tif_file = '/mnt/h/20230620/LTB4-500nM-2.5uL-1_*.tif'
multi_tif = skimage.io.MultiImage(tif_file)
length = len(multi_tif)

for i in range(length):
    image = multi_tif[i][0:200, 155:1300] / 256
    
    csv_filename = f'/mnt/h/20230620/data/nodes20230620_{i}.csv'
    with open(csv_filename, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        detections = []
        for row in reader:
            detections.append(list(map(float, row[1:3])))
        detections = np.array(detections)

    fig, ax = plt.subplots(figsize=(18, 2.5))
    ax.imshow(image, cmap='gray')
    ax.axis("off")
    ax.scatter(detections[:, 1], detections[:, 0], s=20, linewidths=1, marker="x", color="red")
    
    output_filename = f'/mnt/h/20230620/processed/LTB4-500nM-2.5uL-1_{i}.png'
    fig.savefig(output_filename)
    plt.close(fig)
