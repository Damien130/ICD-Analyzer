from PIL import Image
import glob
import random
import os
from concurrent.futures import ThreadPoolExecutor

def extract_random_frame(tiff_files, output_directory):
    # Select a random TIFF file
    tiff_file = random.choice(tiff_files)

    # Open the TIFF file
    tiff = Image.open(tiff_file)

    # Select a random frame
    frame_index = random.randint(0, tiff.n_frames - 1)

    # Go to the selected frame
    tiff.seek(frame_index)

    # Convert to RGB mode if required (some TIFFs are in CMYK)
    if tiff.mode != 'RGB':
        tiff = tiff.convert('RGB')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the frame as a PNG file
    output_path = os.path.join(output_directory, f"random_frame_{frame_index}.png")
    tiff.save(output_path)

    print(f"Saved random frame from {tiff_file}: {output_path}")
# Specify the path to the TIFF file(s) using wildcards
tiff_files = glob.glob('/mnt/d/Benam Lab/20230620/LTB4-500nM-2.5uL-1_*.tif')

# Specify the output directory
output_directory = '/mnt/h/training/png'

num_pictures = 300

with ThreadPoolExecutor(max_workers=16) as executor:
    for _ in range(num_pictures):
        executor.submit(extract_random_frame, tiff_files, output_directory)
