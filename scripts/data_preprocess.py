# 1. We don't need to include metadata.csv yet.
# 2. We first split the sequences into segments of length `seq_len`.
# 3. After splitting, we create an metadata.csv file.
# 4. We can try 10s history + 50s future prediction first.

import argparse
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import multiprocessing
import os
from pathlib import Path
import pandas as pd


def main(args):
    def extract(raw_queue, process_queue, worker_counter):
        while raw_queue.qsize() > 0:
            fn = raw_queue.get()
            if fn == 'Exit':
                with worker_counter.get_lock():
                    worker_counter.value -= 1
                return 0
            else:
                data = np.load(fn + '/track.npz')
                if 'track' not in data.files:
                    print(f"'track' key not found in file {fn}")
                    continue
                data = data['track']
            
            nonneutral_mask = data[-1,:,1] != 16
            subsampled = data[::args.sampling, nonneutral_mask, 0:13]
            subsampled = np.swapaxes(subsampled, 0, 1)
            total_frame = subsampled.shape[1]
            sampled_frame = total_frame - total_frame % args.seq_len
            if sampled_frame // args.seq_len == 0:
                continue
            list_seq = np.split(subsampled[:,:sampled_frame,:], sampled_frame//args.seq_len, axis=1)
            for i, seq in enumerate(list_seq):
                name = fn.split('/')[-1]
                process_queue.put([name + '_' + str(i) + '.npy', seq])
        

    def save(process_queue, process_num, worker_counter, save_dir):
        while worker_counter.value > 0 or process_queue.qsize() > 0:
            fn, obj = process_queue.get()
            if isinstance(obj, str) and obj == "Exit":
                print("Job done, process {} exit".format(process_num))
                return 0
            else:
                np.save(save_dir + fn, obj)
                
    # list all files
    save_dir = args.root + args.dataset_name + '/processed/'
    os.makedirs(save_dir, exist_ok=False)
    raw_queue = multiprocessing.Queue()  # Create shared queue
    files = os.listdir(args.root + args.dataset_name + '/raw/')
    # select 10% as the test set.
    worker_counter = multiprocessing.Value('i', args.num_workers)
    for fn in files:
        raw_queue.put(args.root + args.dataset_name + '/raw/' + fn)
    process_queue = multiprocessing.Queue()  # Create shared queue
    
    # Start worker processes
    worker_processes = [
        multiprocessing.Process(target=extract, args=(raw_queue, process_queue, worker_counter)) for i in range(args.num_workers)
    ]
    for p in worker_processes:
        p.start()

    # Start listener processes
    listener_processes = [
        multiprocessing.Process(target=save, args=(process_queue, i, worker_counter, save_dir)) for i in range(args.num_listeners)
    ]
    for p in listener_processes:
        p.start()

    for p in worker_processes + listener_processes:
        p.join() 

    npz_files = [x for x in Path(save_dir).iterdir() if x.endswith('.npz')]

    metadata = np.array(npz_files)
    df = pd.DataFrame(metadata,columns=['fname'])
    df.to_csv(save_dir + 'metadata.csv', index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess StarCraftMotion dataset')
    parser.add_argument('--root', default="/local/scratch/a/bai116/datasets/")
    parser.add_argument('--sampling', default=24, type=int)
    parser.add_argument('--seq_len', default=10, type=int)
    parser.add_argument('--include_neutral', default=False, action="store_true")
    parser.add_argument('--mp', default=False, action="store_true")
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--num_listeners', default=15, type=int)
    parser.add_argument('--dataset_name', default="StarCraftMotion_v0.2")
    args = parser.parse_args()
    main(args)

