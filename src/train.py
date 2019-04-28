import os
from functools import partial
from datetime import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.callbacks import TensorBoard, ModelCheckpoint

from config import *
from models import unet_2x
from losses import focal_loss
from metrics import mean_iou
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
            write_graph=False,
            write_images=True
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


def stadardize(featurewise_std, batch):
    batch -= batch.mean(axis=(1,2), keepdims=True)
    batch /= featurewise_std
    return batch


def train(model):
    validation_fraction = 0.1
    subset = load_subset(os.path.join(ELEVATION_TIF_DIR, 'sub'), frac=0.2)
    featurewise_std = subset.std()

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
    batch_size = 32
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

    validation_samples = int(len(os.listdir(os.path.join(ELEVATION_TIF_DIR, 'sub'))) * 4 * (validation_fraction))
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

    train_generator = zip(image_train_generator, mask_train_generator)
    validation_generator = zip(image_validation_generator, mask_validation_generator)
    validation_data = next(validation_generator)

    train_samples = int(len(os.listdir(os.path.join(ELEVATION_TIF_DIR, 'sub'))) * 4 * (1-validation_fraction))
    training_history = model.fit_generator(
        train_generator,
        steps_per_epoch=int(train_samples/batch_size),
        epochs=25,
        callbacks=compile_callbacks(),
        validation_data=validation_data
    )

    return model, training_history

if __name__ == "__main__":
    input_img = Input((*IMAGE_SIZE, 1), name='img')
    model = unet_2x(input_img, n_filters=8, dropout=0.05, batchnorm=False, logits=False)
    model.compile(optimizer='Adam', loss=focal_loss, metrics=['accuracy', mean_iou])

    model, history = train(model)