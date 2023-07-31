import imageio
import glob

def assemble_png_to_mov(png_folder, output_path, fps):
    png_files = sorted(glob.glob(png_folder + '/*.png'))  # Get PNG files in numerical order
    writer = imageio.get_writer(output_path, format='ffmpeg', mode='I', fps=fps)

    for png_file in png_files:
        image = imageio.imread(png_file)
        writer.append_data(image)

    writer.close()

# Usage example
png_folder = '/mnt/h/20230620/processed/png_frames'
output_path = '/mnt/h/20230620/processed/animation.mp4'
fps = 30

assemble_png_to_mov(png_folder, output_path, fps)
