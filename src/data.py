import shutil
import os
from urllib.parse import urlparse

import requests
from PIL import Image

from trowel import utils as tile_utils
from config import TERRAIN_BASE_URL

def terrain_tile_url(lon, lat, zoom):
    (X, Y) = tile_utils.lonlat_to_tile(lon, lat, zoom)
    return os.path.join(TERRAIN_BASE_URL, str(zoom), str(X), str(Y)) + '.png'

def url_to_path(root, url, ending):
    path = urlparse(url).path
    no_ending = path.split('.')[-2]
    [z, y, x] = no_ending.split('/')[-3:]
    destination_file = z + '_' + y + '_' + x + ending
    return os.path.join(root, destination_file)

def download_tile(url, destination=None):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        if destination:
            destination_file = url_to_path(destination, url, '.png')
            with open(destination_file, 'wb') as out_file:
                shutil.copyfileobj(r.raw, out_file)
        else:
            return Image.open(r.raw)
    else:
        print('problem downloading file: {}'.format(url))

def decode_elevation(data):
    '''
    decode the elevation from the RGB channels into a single grayscale array
    see https://mapzen.com/documentation/terrain-tiles/formats/

    Arguments:
    data - an NxNx3 numpy array where the last dimension contains the RGB layers,
    i.e. data[:, :, 0] are the red values, etc.

    Returns:
    an NxN numpy float array containing elevation in meters
    '''
    return (data[:, :, 0] * 256 + data[:, :, 1] + data[:, :, 2] / 256) - 32768

