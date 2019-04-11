
import numpy as np
from scipy import signal
from PIL import Image, ImageFilter

GRADIENT_THRESHOLD = 50
CURVATURE_THRESHOLD = 10e3  

def resize(image, size):
    im = Image.fromarray(image)
    im_new = im.resize(size=size, resample=Image.BILINEAR)
    return np.array(im_new)

def erode(image):
    im = Image.fromarray(image)
    im_new = im.filter(ImageFilter.MinFilter(3))
    return np.array(im_new)

def local_maxima(image):
    kernel = np.array([
        [-1, 0, -1],
        [0, 1, 0],
        [-1, 0, -1]
    ])
    kernel = kernel - kernel.mean()
    print(kernel, kernel.sum())
    maxima = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    return maxima - image

def gradient_x(image):
    kernel = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    grad_x = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    return grad_x

def gradient_y(image):
    kernel = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    grad_y = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    return grad_y

def gradient(image):
    grad_x = gradient_x(image)
    grad_y = gradient_y(image)
    return np.sqrt(grad_x**2 + grad_y**2)

def curvature(image):
    return gradient(gradient(image))

def scharr(image):
    kernel = np.array([
        [ -3-3j, 0-10j,  +3 -3j],
        [-10+0j, 0+ 0j, +10 +0j],
        [ -3+3j, 0+10j,  +3 +3j]
    ]) # Gx + j*Gy
    
    grad = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    return np.absolute(grad), np.angle(grad)

def scharr_curvature(image):
    kernel = np.array([
        [ -3-3j, 0-10j,  +3 -3j],
        [-10+0j, 0+ 0j, +10 +0j],
        [ -3+3j, 0+10j,  +3 +3j]
    ]) # Gx + j*Gy
    
    grad = signal.convolve2d(image, kernel, boundary='symm', mode='same')
    curvature = signal.convolve2d(grad, kernel, boundary='symm', mode='same')
    return np.absolute(curvature)

def ridges(elevation):
    candidate_ridges = []
    total_gradient = gradient(elevation)
    curvature = resize(scharr_curvature(resize(elevation, size=(64, 64))), size=(256, 256))

    for size in [(256, 256), (128, 128), (64, 64)]:
        elevation_reduced = resize(elevation, size=size)

        grad_x = gradient_x(elevation_reduced)
        curv_x = gradient_x(grad_x > 0)

        grad_y = gradient_y(elevation_reduced)
        curv_y = gradient_y(grad_y > 0)

        curv_xy = (curv_x<0) + (curv_y<0)

        original_size = resize(curv_xy.astype(np.int8), (256, 256))
        candidate_ridges.append(original_size)

    candidate_ridges = np.stack(candidate_ridges + [curvature > CURVATURE_THRESHOLD], axis=2)
    return np.all(candidate_ridges, axis=2)