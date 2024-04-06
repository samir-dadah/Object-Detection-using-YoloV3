import os
import cv2 as cv
from matplotlib import pyplot as plt
images = cv.imread('/content/image.jpg')

from numpy import expand_dims
from keras.preprocessing.image import load_img, img_to_array
 
# load and prepare an image
def load_image_pixels(filename, shape):
     
    # load the image to get its shape
    image = load_img(filename)
    width, height = image.size
    
    # load the image with the required size
    image = load_img(filename, target_size=shape)
    
    # convert to numpy array
    image = img_to_array(image)
    
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0
    
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height
