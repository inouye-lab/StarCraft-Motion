import os
import numpy as np
import pandas as pd

dir = "./data/starcraft-motion-dataset"
filtered_names = []
for filename in os.listdir(dir):
    if filename.endswith('.npz'):
        base_name = filename.split('_')[0]  # Split and take the first part
        if base_name not in filtered_names:
            filtered_names.append(base_name)

test_set = np.random.choice(filtered_names, size=int(len(filtered_names)*0.1), replace=False)

meta = []
for filename in os.listdir(dir):
    if filename.endswith('.npz'):
        base_name = filename.split('_')[0]  # Split and take the first part
        if base_name in test_set:
            meta.append([filename, 1])
        else:
            meta.append([filename, 0])
metadata = np.array(meta)

df = pd.DataFrame(metadata, columns=['fname', 'is_test'])
df.to_csv(dir + 'metadata.csv', index=True)
