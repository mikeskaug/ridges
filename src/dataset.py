import os

from PIL import Image
import numpy as np

from trowel import utils

from src.data import url_to_path, download_tile, decode_elevation
from src.config import TERRAIN_BASE_URL, ELEVATION_PNG_DIR, DEFAULT_ZOOM
from src.transforms import ridges


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