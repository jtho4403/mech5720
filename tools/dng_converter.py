import rawpy
import imageio.v3 as iio
import numpy as np 

DATA_PATH = "data/lab2/captured/"
OUT_PATH = "data/lab2/captured_png/" 

from pathlib import Path
import rawpy
import imageio.v3 as iio

# Set the path to your folder
folder_path = Path(DATA_PATH)
out_path = Path(OUT_PATH)
out_path.mkdir(parents=True, exist_ok=True)

# Loop through all DNG files in that specific folder
for file_path in folder_path.glob("*.dng"):
    print(f"Converting: {file_path}")
    
    filename  = file_path.stem
    output_path = out_path / (filename + ".png")

    # Load the DNG file
    with rawpy.imread(str(file_path)) as raw:
        # Postprocess the raw data into a viewable sRGB image array
        rgb = raw.postprocess(use_camera_wb=True, half_size=True)
        
    # Save the array directly as a PNG file
    iio.imwrite(output_path, rgb) 
