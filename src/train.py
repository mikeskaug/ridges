import os
from functools import partial
from datetime import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

from config import *
from models import unet_2x, LFE, stacked_multi_scale
from losses import focal_loss, balanced_cross_entropy, dice_loss, bce_plus_dice
from metrics import iou, dice_coefficient
from dataset import load_subset


def compile_callbacks(
        logs_dir=os.path.join(LOGS_DIR, datetime.now().isoformat(timespec='minutes')), 
        checkpoint_dir=os.path.join(CHECKPOINT_DIR, datetime.now().isoformat(timespec='minutes'))
    ):
    if not os.path.isdir(logs_dir):
        os.mkdir(logs_dir)
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    return [
        TensorBoard(
            log_dir=os.path.join(logs_dir),
            histogram_freq=1,
            update_freq='epoch',
            write_graph=False,
            write_images=True,
            profile_batch=0
        ),
        # EarlyStopping(patience=10, verbose=1),
        # ReduceLROnPlateau(factor=0.1, patience=3, min_lr=0.00001, verbose=1),
        ModelCheckpoint(    
            os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
            verbose=1, 
            save_best_only=True, 
            save_weights_only=True
        )
    
    ]


def stadardize(featurewise_std, image):
    image -= image.mean()
    image /= featurewise_std
    return image


def train(model):
    validation_fraction = 0.05
    subset = load_subset(os.path.join(ELEVATION_TIF_DIR, 'sub'), frac=0.2)
    featurewise_std = subset.std()

    augmentation_factor = 3 # the additional factor of training samples obtained via augmentation
    image_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=partial(stadardize, featurewise_std),
        validation_split=validation_fraction
    )
    mask_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=validation_fraction,
        rescale=1/255
    )

    seed = 1
    batch_size = 16
    image_train_generator = image_datagen.flow_from_directory(
        ELEVATION_TIF_DIR,
        color_mode='grayscale',
        class_mode=None,
        seed=seed,
        batch_size=batch_size,
        subset='training'
    )

    mask_train_generator = mask_datagen.flow_from_directory(
        MASK_TIF_DIR,
        color_mode='grayscale',
        class_mode=None,
        seed=seed,
        batch_size=batch_size,
        subset='training'
    )

    validation_samples = int(len(os.listdir(os.path.join(ELEVATION_TIF_DIR, 'sub'))) * (validation_fraction))
    image_validation_generator = image_datagen.flow_from_directory(
        ELEVATION_TIF_DIR,
        color_mode='grayscale',
        class_mode=None,
        seed=seed,
        batch_size=validation_samples,
        subset='validation'
    )

    mask_validation_generator = mask_datagen.flow_from_directory(
        MASK_TIF_DIR,
        color_mode='grayscale',
        class_mode=None,
        seed=seed,
        batch_size=validation_samples,
        subset='validation'
    )

    train_generator = (pair for pair in zip(image_train_generator, mask_train_generator))
    validation_generator = zip(image_validation_generator, mask_validation_generator)
    validation_data = next(validation_generator)

    train_samples = int(len(os.listdir(os.path.join(ELEVATION_TIF_DIR, 'sub'))) * (1-validation_fraction) * augmentation_factor)
    training_history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(train_samples/batch_size),
        epochs=50,
        callbacks=compile_callbacks(),
        validation_data=validation_data
    )

    return model, training_history


if __name__ == "__main__":
    input_img = Input((*IMAGE_SIZE, 1), name='img')
    # model = unet_2x(input_img, n_filters=8, dropout=0.0, batchnorm=False, logits=False)
    # model = LFE(input_img, n_filters=8, batchnorm=False, logits=False)
    model = stacked_multi_scale(input_img, n_filters=16, batchnorm=False, logits=False)
    model.compile(optimizer='Adam', loss=bce_plus_dice, metrics=['accuracy', dice_coefficient])

    model, history = train(model)