import soundfile as snd
import os
from tqdm import tqdm

for i in ['train', 'test', 'valid']:
    tsv = i+'.tsv'
    lines = open(tsv, 'r').readlines()
    path = lines[0].strip()
    length = 0
    for fname in tqdm(lines[1:]):
        length_ = fname.split('\t')[1]
        length += int(length_)/16000
    print(i,length/3600)
