from collections import defaultdict
import numpy as np
import csv
from mask_functions import rle2mask
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def show_img(pixels, fname):
    image = cv2.imread(f'images/processed/train/{fname}.png', cv2.IMREAD_GRAYSCALE)
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(pixels.reshape(pixels.shape[0:2]), cmap=plt.get_cmap('bone'))
    ax[1].imshow(image, cmap=plt.get_cmap('bone'))
    fig.show()


def make_mask(rles):
    mask = np.zeros((1024, 1024), dtype=np.float)
    for rle in (x for x in rles if x != [-1]):
        mask = np.ma.mask_or(mask, rle2mask(rle, 1024, 1024).squeeze()).astype(np.float)
    return np.swapaxes(mask, 0, 1) * 255


labels = defaultdict(list)
with open('train-rle.csv') as fh:
    reader = csv.DictReader(fh)
    for row in reader:
        rle = [int(x) for x in row['EncodedPixels'].split()]
        labels[row['ImageId']].append(rle)

prog = tqdm(list(labels.items()))
zeros = 0
ones = 0
for fname, rles in prog:
    m = make_mask(rles).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(m, kernel, iterations=2)
    total = np.count_nonzero(dilated)
    ones += (dilated == 255).sum()
    zeros += (dilated == 0).sum()

    # cv2.imwrite(f'images/processed/masks-dilated/0/{fname}.png', dilated)

print("0s", zeros, "1s", ones, "1s %:", float(ones) / float(ones + zeros),"0s %:", float(zeros)/float(ones+zeros))
