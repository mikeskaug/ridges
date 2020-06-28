import os

from PIL import Image
import numpy as np
from tensorflow.keras.utils import Sequence

from trowel import utils

from data import url_to_path, download_tile, decode_elevation
from config import TERRAIN_BASE_URL, ELEVATION_PNG_DIR, DEFAULT_ZOOM, IMAGE_SIZE
from transforms import ridges


def harvest_tiles(bounds, zoom=DEFAULT_ZOOM, out_dir=ELEVATION_PNG_DIR):
    '''
    Download all the terrain tiles covering the bounding box defined by bounds and save to out_dir
    '''

    tiles = utils.bbox_to_tiles(*bounds, zoom)
    for tile in tiles:
        url = os.path.join(TERRAIN_BASE_URL, str(tile[0]), str(tile[1]), str(tile[2])) + '.png'
        download_tile(url, destination=out_dir)


def png_to_tif(png_dir, tif_dir):
    '''
    Convert the RGB elevation tiles into grayscale tif files
    '''
    for fl in os.listdir(png_dir):
        png = Image.open(os.path.join(png_dir, fl))

        elevation = decode_elevation(np.array(png))

        grayscale = Image.fromarray(elevation)
        grayscale.save(os.path.join(tif_dir, fl.replace('.png', '.tif')))


def create_masks(tif_dir, mask_dir):
    for fl in os.listdir(tif_dir):
        tif = Image.open(os.path.join(tif_dir, fl))
        elevation = np.array(tif)
        ridge_mask = ridges(elevation)

        mask = Image.fromarray(ridge_mask.astype(np.int8) * 256).convert('L')
        mask.save(os.path.join(mask_dir, fl))


def standardize_batch(featurewise_std, batch):
    batch -= batch.mean(axis=(1,2), keepdims=True)
    batch /= featurewise_std

    return batch
    

def load_subset(data_dir, N=None, frac=None, seed=1):
    '''
    Load a random subset of the images in a directory and return as an Nx256x256x1 numpy array
    '''
    np.random.seed(seed=seed)
    files = os.listdir(data_dir)
    if N:
        subset_files = np.random.choice(files, size=N, replace=False)
    elif frac:
        subset_files = np.random.choice(files, size=int(frac*len(files)), replace=False)
    
    data = []
    for fl in subset_files:
        im = Image.open(os.path.join(data_dir, fl))
        data.append(np.array(im))

    return np.expand_dims(np.stack(data), axis=3)


class CustomImageDataGenerator(Sequence):
    '''
    Custom image data generator that avoid some limitations of Keras ImageDataGenerator
    '''
    def __init__(self, image_path, mask_path, image_filenames,
                to_fit=True, batch_size=32, augment=True, standardize_batch=lambda x: x,
                seed=1, shuffle=True, rescale_x=1, rescale_y=1, n_outputs=1):
        '''Initialization
        :param list_IDs: list of all 'label' ids to use in the generator
        :param labels: list of image labels (file names)
        :param image_path: path to images location
        :param mask_path: path to masks location
        :param to_fit: True to return X and y, False to return X only
        :param batch_size: batch size at each iteration
        :param dim: tuple indicating image dimension
        :param n_channels: number of image channels
        :param n_classes: number of output masks
        :param shuffle: True to shuffle label indexes after every epoch
        '''
        np.random.seed(seed=seed)
        self.dim = IMAGE_SIZE
        self.augment = augment
        self.augment_factor = 8 if augment else 1
        self.image_files = image_filenames * self.augment_factor
        self.image_idxs = np.arange(len(self.image_files))
        self.image_path = image_path
        self.mask_path = mask_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.n_channels = 1
        self.shape = (256, 256)
        self.rescale_x = rescale_x
        self.rescale_y = rescale_y
        self._standardize_batch = standardize_batch
        self.n_outputs = n_outputs
        self.on_epoch_end()

    def __len__(self):
        '''
        Return the number of batches per epoch
        '''
        return int(np.floor(len(self.image_idxs)/ self.batch_size))

    def __getitem__(self, index):
        '''
        Generate a batch of training samples and target masks
        '''
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_IDs_temp = [self.image_idxs[k] for k in indexes]

        X = self._generate_X(list_IDs_temp)
        X = self._standardize_batch(X)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            if self.augment:
                X, y = self._augment(X, y)
            return (X, (y,)*self.n_outputs)
        else:
            return X


    def on_epoch_end(self):
        '''Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.image_idxs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def _generate_X(self, list_IDs_temp):
        '''
        Return an rank 4 array containing a batch of image examples (batch, height, width, 1)
        '''
        X = np.empty((self.batch_size, *self.shape, self.n_channels))

        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self._load_image(os.path.join(self.image_path, self.image_files[ID]))[:, :, np.newaxis]

        return X * self.rescale_x


    def _generate_y(self, list_IDs_temp):
        '''Generates data containing batch_size masks
        :param list_IDs_temp: list of label ids to load
        :return: batch if masks
        '''
        y = np.empty((self.batch_size, *self.shape, 1), dtype=int)

        for i, ID in enumerate(list_IDs_temp):
            y[i,] = self._load_image(os.path.join(self.mask_path, self.image_files[ID]))[:, :, np.newaxis]

        return y * self.rescale_y


    def _load_image(self, image_path):
        '''
        Return a grayscale image as a 2D numpy array
        '''
        im = Image.open(image_path)
        data = np.array(im)
        return data

    
    def _augment(self, X, y):
        '''
        Randomly flip and rotate each image and target in a batch
        '''
        for idx in np.arange(X.shape[0]):
            if np.random.rand() > 0.5:
                X[idx,] = np.flipud(X[idx,])
                y[idx,] = np.flipud(y[idx,])
            
            rot_steps = np.random.choice([0, 1, 2, 3])
            X[idx,] = np.rot90(X[idx,], k=rot_steps)
            y[idx,] = np.rot90(y[idx,], k=rot_steps)

        return X, y