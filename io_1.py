from PIL import Image

from skimage import io
from skimage.color import rgb2gray, rgba2rgb

from classes import Image
from config import Config
import cv2

config = Config()

def importImage(path):
    """ Imports a given image and converts it into two copies, one as grayscale, one as a binarized skeletonized copy.

    :param path: str: File path of the image.
    :return image: ndarray: Grayscale conversion of imported image.
    :return binaryImage: ndarray: Binarized Skeletonized copy of imported image.
    """
    image = cv2.imread(path)
    if len(image.shape) == 2:
        image = Image(image, path)
        return image
    else:

        dim3 = image.shape[2]
        if dim3 == 4:
            image = rgba2rgb(image)
            image = rgb2gray(image)

            image = Image(image, path)
            return image
        elif dim3 == 3:
            image = rgb2gray(image)
            image = Image(image, path)
            return image
        else:
            image = Image(image, path)
            return image