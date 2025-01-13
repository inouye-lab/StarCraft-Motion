import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import shutil

di = "/local/scratch/a/bai116/datasets/StarCraftMotion_v0.2/raw/"

filtered_names = []
nonpz_files = []
for filename in tqdm(os.listdir(di)):
    for f in os.listdir(di + filename):
        print(f) 
        if f.endswith('.npz'):
            try:
                t = np.load(di + filename + '/' + f)
                print(t['track'].shape)
            except:
                nonpz_files.append(filename)
            else:
                filtered_names.append(filename)
            finally:
                break
    else:
        nonpz_files.append(filename)
print(len(filtered_names))
print(len(nonpz_files))
print("----")
for nonpz in tqdm(nonpz_files):
    print(nonpz)
    # shutil.rmtree(dir + '/' + nonpz)
