"""
Created by Artur - 29-March-2022

Script to remove corrupted/blank images from the cats dataset
"""

import os
from tqdm import tqdm
for root, dirs, files in tqdm(os.walk(".", topdown=False)):
   for name in files:
      file = os.path.join(root, name)
      fs = os.path.getsize(file)
      if fs < 1000:
        mvfile = os.path.join("/scratch/arturao/GANSketching_old/data/image/cat_old", name)
        os.rename(file, mvfile)
        print(f"{name} - {fs} bytes")
    