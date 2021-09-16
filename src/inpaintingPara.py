import numpy as np
import imageio
import Poisson as poi
import matplotlib.pyplot as plt
from scipy import ndimage

def get_rand_mask(mask):
    rand_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=bool)
    for y in range(mask.shape[0]):
        for x in range(mask.shape[1]):
            false_n = False
            true_n = False
            if mask[y, x]:
                if mask[y, x - 1] or mask[y, x + 1] or mask[y - 1, x] or mask[y + 1, x]:
                    true_n = True
                if mask[y, x - 1] == False or mask[y, x + 1] == False or mask[y - 1, x] == False or mask[
                    y + 1, x] == False:
                    false_n = True
                if true_n and false_n:
                    rand_mask[y, x] = True

    return rand_mask





def inpaint(img, n, x, y):


    if len(img.shape) == 3:  #if statement for handling color picure
            img = np.sum(img.astype(float), 2) / (3 * 255)

    mask = np.zeros((img.shape[0], img.shape[1], 2), dtype=bool)
    err_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

    err_x1 = x[0]+10
    err_y1 = y[0]+10
    err_x2 = x[1]-10
    err_y2 = y[1]-10

    err_mask[err_y1:err_y2, err_x1:err_x2] = True
    mask[y[0]:y[1], x[0]:x[1], 0] = True
    mask[:,:,1] = get_rand_mask(mask[:,:,0])

    img[err_mask[:,:]] = 0

    edit_img = poi.poisson(img, n, rand='dericle', mask=mask)


    return edit_img