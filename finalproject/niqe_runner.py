

import cv2

from niqe import niqe
from pique import pique

def runner(path):
    im = cv2.imread(path)
    if im is None:
        print("Failed to read image file")
        raise BrokenPipeError

    else:
        score = niqe(im)
        return("{}".format(score))

def pique_runner(path):
    im = cv2.imread(path)
    if im is None:
        print("Failed to read image file")
        raise BrokenPipeError

    else:
        score, _, _, _ = pique(im)
        return("{}".format(score))