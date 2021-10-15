import os
import csv
import shutil

if __name__ == "__main__":
    labels = {}
    cfolder = "images/processed/single_class"
    if os.path.exists(cfolder):
        shutil.rmtree(cfolder)
    os.makedirs(f"{cfolder}/0", exist_ok=True)
    os.makedirs(f"{cfolder}/1", exist_ok=True)
    with open('train-rle.csv', 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            cls = int(row['EncodedPixels'].strip() != "-1")
            a = os.path.abspath(f"images/processed/train/{row['ImageId']}.png")
            b = os.path.abspath(f"{cfolder}/0/{row['ImageId']}.png")
            if os.path.exists(a) and not os.path.exists(b):
                print(f"Creating link\n{b}->\n{a}")
                os.symlink(a, b)
