import os
import sys


module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

TERRAIN_BASE_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium'
DEFAULT_ZOOM = 12

COLORADO_BOUNDS = (-109.00634, 37.02886, -102.09594, 41.0)

ELEVATION_PNG_DIR = f'{os.environ["RIDGES_ROOT"]}/data/high_ridge_terrain/Colorado/train/png'
ELEVATION_TIF_DIR = f'{os.environ["RIDGES_ROOT"]}/data/high_ridge_terrain/Colorado/train/tif/sub'
MASK_TIF_DIR = f'{os.environ["RIDGES_ROOT"]}/data/high_ridge_terrain/Colorado/train/mask/sub'
LOGS_DIR = f'{os.environ["RIDGES_ROOT"]}/output/logs'
CHECKPOINT_DIR = f'{os.environ["RIDGES_ROOT"]}/output/checkpoint'

IMAGE_SIZE = (256, 256)

