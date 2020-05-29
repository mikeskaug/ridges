import os
from functools import partial
from datetime import datetime

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from config import *
from models import unet_2x, LFE, stacked_multi_scale, HED
from losses import focal_loss, balanced_cross_entropy, dice_loss, bce_plus_dice, per_sample_balanced_cross_entropy
from metrics import iou, dice_coefficient
from dataset import load_subset, CustomImageDataGenerator


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
            write_images=False,
            profile_batch=0
        ),
        # EarlyStopping(patience=10, verbose=1),
        # ReduceLROnPlateau(
        #     monitor='loss',
        #     factor=0.1, 
        #     patience=5, 
        #     min_lr=0.00001, 
        #     verbose=1,
        #     cooldown=10
        # ),
        ModelCheckpoint(
            os.path.join(checkpoint_dir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5'), 
            verbose=1,
            save_freq='epoch',
            save_best_only=False, 
            save_weights_only=True
        )
    
    ]


def standardize_batch(featurewise_std, batch):
    batch -= batch.mean(axis=(1,2), keepdims=True)
    batch /= featurewise_std

    return batch


def train(model):
    validation_fraction = 0.05
    batch_size = 8
    subset = load_subset(ELEVATION_TIF_DIR, frac=0.2)
    featurewise_std = subset.std()

    files = os.listdir(ELEVATION_TIF_DIR)
    np.random.shuffle(files)
    train_files = files[:int(len(files)*(1-validation_fraction))]
    validation_files = files[int(len(files)*(1-validation_fraction)):]

    training_generator = CustomImageDataGenerator(
        ELEVATION_TIF_DIR, 
        MASK_TIF_DIR, 
        train_files, 
        batch_size=batch_size, 
        standardize_batch=partial(standardize_batch, featurewise_std), 
        rescale_y=1/255,
        n_outputs=5
    )

    validation_generator = CustomImageDataGenerator(
        ELEVATION_TIF_DIR, 
        MASK_TIF_DIR, 
        validation_files, 
        batch_size=len(validation_files)*8, 
        standardize_batch=partial(standardize_batch, featurewise_std), 
        rescale_y=1/255,
        n_outputs=5
    )
    validation_data = validation_generator.__getitem__(0)

    training_history = model.fit(
        training_generator,
        steps_per_epoch=len(training_generator),
        epochs=50,
        callbacks=compile_callbacks(),
        shuffle=False,
        validation_data=validation_data
    )

    return model, training_history


if __name__ == "__main__":
    input_img = Input((*IMAGE_SIZE, 1), name='img')
   
    model = HED(input_img)
    model.compile(loss={'o1': balanced_cross_entropy,
                        'o2': balanced_cross_entropy,
                        'o3': balanced_cross_entropy,
                        'o4': balanced_cross_entropy,
                        'ofuse': balanced_cross_entropy,
                        },
                  metrics={'ofuse': ['accuracy', dice_coefficient]},
                  optimizer=Adam(learning_rate=0.001))
    model, history = train(model)