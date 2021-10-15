import cv2
import pydicom
from multiprocessing import Pool
from pathlib import Path
import os
from datetime import datetime
import tqdm
import pandas as pd


def do_img(fname, outfolder):
    pixels = pydicom.read_file(str(fname)).pixel_array
    cv2.imwrite(f"{outfolder}/{fname.stem}.png", pixels)


if __name__ == "__main__":
    pool = Pool()

    start = datetime.now()
    paths = []
    train_labels = {}
    labels = set(pd.read_csv('train-rle.csv').ImageId)
    for t in ("test", "train"):
        outfolder = f'images/processed/{t}'
        os.makedirs(outfolder, exist_ok=True)
        paths.extend((x, outfolder) for x in Path(f'images/dicom-images-{t}').glob('**/*.dcm') if t == 'test' or x.stem in labels)

    prog = tqdm.tqdm(desc='Loading Images', total=len(paths))

    for p in paths:
        pool.apply_async(do_img, args=p, callback=prog.update, error_callback=print)
    pool.close()
    pool.join()
