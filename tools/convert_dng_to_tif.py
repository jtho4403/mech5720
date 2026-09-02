import rawpy
import imageio
import glob
import os
import re

SRC_DIR = "/Users/jackthompson/MECH5720/codeRepo/mech5720/data/lab2/captured"

for dng_path in glob.glob(os.path.join(SRC_DIR, "*.dng")):
    with rawpy.imread(dng_path) as raw:
        rgb = raw.postprocess(
            gamma=(1, 1),          # linear output, no display gamma
            no_auto_bright=True,   # don't rescale exposure
            use_camera_wb=False,   # don't apply white balance
            user_wb=[1, 1, 1, 1],  # force unity gains on all channels
            output_bps=16,         # keep full precision
            highlight_mode=rawpy.HighlightMode.Clip,  # no highlight recovery
            demosaic_algorithm=rawpy.DemosaicAlgorithm.AHD,
        )
    out_path = os.path.splitext(dng_path)[0] + ".tif"
    imageio.imwrite(out_path, rgb)
    print(os.path.basename(dng_path), "->", os.path.basename(out_path))
