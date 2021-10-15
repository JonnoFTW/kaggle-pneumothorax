import os
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import csv
from unet_model import UnetModel
from keras_preprocessing.image import ImageDataGenerator, DirectoryIterator
import numpy as np
from generate_augmented import get_generator
from mask_functions import mask2rle, rle2mask
from LRFinder import LRFinder
from unetpp_model import UnetPlusPlusModel
from unet_backbone import UnetBackBoneModel

INPUT_SIZE = (256, 256)
batch_size = 8
RESIZE = False


def show_img(pixels):
    plt.imshow(pixels.reshape, cmap=plt.get_cmap('bone'))
    plt.show()


def get_testing_generator(img_gen: DirectoryIterator):
    gen = ImageDataGenerator(
        rescale=1 / 255.
        # featurewise_center=True,
        # featurewise_std_normalization=True,
    )
    # gen.std = img_gen.image_data_generator.std
    # gen.mean = img_gen.image_data_generator.mean
    return gen.flow_from_directory(
        directory='images/processed/test',
        batch_size=batch_size,
        target_size=INPUT_SIZE,
        color_mode='rgb',
        class_mode=None
    )


def make_submission_predictions(model, img_gen):
    """
    Use a model to generate boxed predictions
    :param model:
    :return:
    """
    # read in the submissions file
    # those with multiple lines, expect multiple annotations
    test_labels = defaultdict(list)
    test_images = get_testing_generator(img_gen)
    print("Making Predictions")
    start = datetime.now()
    with open(f'submissions/submissions_{int(datetime.now().timestamp())}.csv', 'w') as fh:
        writer = csv.DictWriter(fh, fieldnames=['ImageId', 'EncodedPixels'])
        writer.writeheader()
        for batch in test_images:
            idx = (test_images.batch_index - 1) * test_images.batch_size
            fnames = test_images.filenames[idx: idx + test_images.batch_size]
            # make predictions on the current batch
            preds = model.predict(batch)
            # preds is an array of
            for pred_idx, pred in enumerate(preds):
                try:
                    fname = fnames[pred_idx]
                except IndexError:
                    continue
                # turn pred into a series of masks

                print(f"Predicting {Path(fname).stem}")
                # TODO get the bounding boxes out of the preds
                writer.writerow({
                    'ImageId': Path(fname).stem,
                    'EncodedPixels': " ".join(["-1"])  # pred
                })
                test_labels[fname].extend(preds)
    print(f"Took: {datetime.now() - start}")
    return test_labels


def get_model(fname=None) -> UnetModel:
    _model = UnetModel(input_size=INPUT_SIZE + (1,))
    if fname and os.path.exists(fname):
        print(f"Reusing existing model {fname}")
        _model.load(fname)
    else:
        print("Creating new model")
        _model.build()
    return _model


def show_batch(imgs, preds, masks=None):
    # if masks is an array, draw it over the imgs
    if masks:
        for idx, m in enumerate(masks):
            imgs[idx] = imgs[idx]
    fig, axes = plt.subplots(preds.shape[0], 2)
    axes[0, 0].set_title("Input")
    axes[0, 1].set_title("Prediction")
    for idx, x in enumerate(zip(imgs, preds)):
        i, p = x
        axes[idx, 0].imshow(i.reshape(i.shape[0:2]), cmap=plt.get_cmap('bone'))
        axes[idx, 1].imshow(p.reshape(np.round(p.shape[0:2])), cmap=plt.get_cmap('bone'))
    fig.show()


def to_it(gen1, gen2):
    while True:
        yield next(gen1), next(gen2)


if __name__ == "__main__":

    labels = defaultdict(list)
    with open('train-rle.csv', 'r') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels[row['ImageId']].append([int(x) for x in row['EncodedPixels'].strip().split()])

    images_folder = 'images/processed/single_class'
    masks_folder = 'images/processed/masks-dilated'
    val_split = 0.15
    image_training_generator, mask_training_generator, image_validation_generator, mask_validation_generator = get_generator(
        masks_folder=masks_folder,
        images_folder=images_folder,
        batch_size=batch_size,
        input_size=INPUT_SIZE,
        validation_split=val_split
    )

    train_iterator = to_it(image_training_generator, mask_training_generator)
    validation_iterator = to_it(image_validation_generator, mask_validation_generator)
    # model = get_model(None)
    model = UnetPlusPlusModel(INPUT_SIZE)
    model.build()
    lrf = LRFinder(model.model)
    validation_samples = image_validation_generator.samples
    train_samples = image_training_generator.samples
    lrf.find_generator(train_iterator, 0.00001, .1, 3, train_samples // batch_size)
    # model.set_lr(lrf.get_best_lr_exp_weighted())
    # model.fit(train_iterator, batch_size, train_samples, validation_samples, validation_iterator, epochs=100)

    # model = get_model('models/unet_1562827607.h5')
    # model.fit(train_iterator, batch_size, train_samples, epochs=1)
    # model.save()
    # model = get_model(None)
    # model = get_model('models/unet_tversky_1562820133.91342.hdf5')
    # lrf = LRFinder(model.model)
    # lrf.find_generator(train_iterator, 0.0001, 1, 1, train_samples//batch_size)
    # new_lr = lrf.get_best_lr(1)
    # print("adjust learning rate to", new_lr)
    # lrf.plot_loss()
    # lrf.plot_loss_change()
    # model.set_lr(new_lr)
    # model.fit(train_iterator, batch_size, train_samples, epochs=10)
    # model.save()
    # test_labels = make_submission_predictions(model, image_generator)
