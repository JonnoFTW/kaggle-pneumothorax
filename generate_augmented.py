import numpy as np
from pathlib import Path
from keras_preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_folder_for_train(folder, input_size):
    paths = np.random.choice(list(Path(folder).glob('**/*.png')), 8000)
    shape = (len(paths),) + input_size + (1,)

    images = np.empty(shape, dtype=np.uint8)
    prog = tqdm(total=len(paths), desc='Image')
    for idx, p in enumerate(paths):
        images[idx] = cv2.resize(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE), input_size).reshape(input_size + (1,))
        prog.update()
    return images


def show_img_and_mask(img, mask):
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.squeeze(), cmap=plt.get_cmap('bone'))
    ax[1].imshow(mask.squeeze(), cmap=plt.get_cmap('bone'))
    fig.show()


def get_fnames(it):
    idx = (it.batch_index - 1) * it.batch_size
    return [it.filenames[it.index_array[i]] for i in range(idx, idx + it.batch_size)]


def get_generator(masks_folder, images_folder, batch_size=2, seed=42069, input_size=(512, 512),validation_split=0.15):
    # we create two instances with the same arguments
    data_gen_args = dict(
        # featurewise_center=True,
        # featurewise_std_normalization=True,
        rescale=1./255,
        # horizontal_flip=True,
        validation_split=validation_split,
        # rotation_range=5,
        # width_shift_range=0.15,
        # horizontal_flip=True,
        # height_shift_range=0.15,
        # fill_mode='constant',
        # cval=0,
        # zoom_range=0.15
    )
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    # images = load_folder_for_train(images_folder, input_size)
    # image_datagen.fit(images, augment=True, seed=seed)
    # del images
    #
    # masks = load_folder_for_train(masks_folder, input_size)
    # mask_datagen.fit(masks, augment=True, seed=seed)
    # del masks

    print("\nTraining Images")
    image_generator = image_datagen.flow_from_directory(
        images_folder,
        class_mode=None,
        target_size=input_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        subset='training'
    )

    print("Training Masks")
    mask_generator = mask_datagen.flow_from_directory(
        masks_folder,
        class_mode=None,
        target_size=input_size,
        shuffle=True,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed,
        subset='training'
    )
    print("Validation Images")
    image_validation_generator = image_datagen.flow_from_directory(
        images_folder,
        class_mode=None,
        target_size=input_size,
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        subset='validation')
    print("Validation Masks")
    mask_validation_generator = mask_datagen.flow_from_directory(
        masks_folder,
        class_mode=None,
        target_size=input_size,
        shuffle=True,
        color_mode='grayscale',
        batch_size=batch_size,
        seed=seed,
        subset='validation')

    # combine generators into one which yields image and masks
    return image_generator, mask_generator, image_validation_generator, mask_validation_generator


def main():
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 56
    batch_size = 2
    images_folder = 'images/processed/single_class'
    masks_folder = 'images/processed/masks'
    input_size = (512, 512)

    image_generator, mask_generator,  image_validation_generator, mask_validation_generator = get_generator(masks_folder, images_folder, batch_size, seed,
                                                    input_size)
    train_gen = zip(image_generator, mask_generator)
    for b in range(10):
        batch = next(train_gen)
        print("Batch", image_generator.batch_index)
        print("\tIMAGES", get_fnames(image_generator))
        print("\tMASKS ", get_fnames(mask_generator))
        for i in range(image_generator.batch_size):
            show_img_and_mask(batch[0][i], batch[1][i])


if __name__ == "__main__":
    main()
