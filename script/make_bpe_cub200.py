import os
from tqdm import tqdm

src_dir = 'dataset/cub200'
files = [f for f in os.listdir(src_dir) if f.endswith('.txt')]


with open('dalle_pytorch/data/cub200.txt', 'w') as fout:
    for fn in tqdm(files):
        with open(os.path.join(src_dir, fn), 'r') as fin:
            lines = fin.readlines()
        for line in lines:
            fout.write(line)
