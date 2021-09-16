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


iter = 50

img = imageio.imread('../hdr-bilder/Balls/Balls_00064.png')
img = np.sum(img.astype(float), 2) / (3 * 255)
print(img.shape)

mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
err_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)

"""
np.random.seed(999)
x, y = (32*np.random.random((2, 210))).astype(np.int)
img[x, y] = 1
"""

x1 = int(input("point 1 x:"))
y1 = int(input("point 1 y:"))
x2 = int(input("point 2 x:"))
y2 = int(input("point 2 y:"))

err_x1 = x1+10
err_y1 = y1+10
err_x2 = x2-10
err_y2 = y2-10

err_mask[err_y1:err_y2, err_x1:err_x2] = True

plt.imshow(err_mask[:,:], plt.cm.gray)
plt.show()

mask[y1:y2, x1:x2] = True

plt.imshow(mask[:,:], plt.cm.gray)
plt.show()

img[err_mask[:,:]] = 0

plt.imshow(img[:,:], plt.cm.gray)
plt.show()


edit_img = poi.poisson(img, iter, mask=mask)


"""
def poisson(u_0, n, h=0, lam=None, rand='dericle'):

    d_x = 1
    d_t = 0.25
    alpha = d_t / (d_x ** 2)

    u = np.zeros((u_0.shape[0], u_0.shape[1], n+1))

    u[:, :, 0] = u_0


    for i in range(n):
        
        if lam is not None:
            h = lam * (u[1:-1, 1:-1, i] - u[1:-1, 1:-1, 0])

                img_mask er en boolean array hvor 1(true) er i indexen hvor img har hull
                for loopene går igjennom arrayene og blurrer vis verdien er 1
                 
       
        for y in range(img.shape[1]-1):
            for x in range(img.shape[0]-1):
                if img_mask[x,y] == True:
                    #u[1:-1, 1:-1, i + 1] = u[1:-1, 1:-1, i] + alpha * (u[:-2, 1:-1, i] + u[2:, 1:-1, i] + u[1:-1, :-2, i] + u[1:-1, 2:, i] - (4 * u[1:-1, 1:-1, i])) - d_t * h
                    u[x, y, i + 1] = u[x, y, i] + alpha * (u[x-1, y, i] + u[x+1, y, i] + u[x, y-1, i] + u[x, y+1, i] - (4 * u[x, y, i])) - d_t * h
        
        if rand == 'dericle':
            u[:, 0, i + 1] = u[:, 0, i]
            u[:, -1, i + 1] = u[:, -1, i]
            u[0, :, i + 1] = u[0, :, i]
            u[-1, :, i + 1] = u[-1, :, i]
        elif rand == 'neuman':
            u[0, 0, i+1] = u[0, 0, i] + alpha * ((2 * u[0, 1, i]) +  (2 * u[1, 0, i]) - (4 * u[0, 0, i])) - d_t * h
            u[-1, 0, i + 1] = u[-1, 0, i] + alpha * ((2 * u[-2, 0, i]) + (2 * u[-1, 1, i]) - (4 * u[-1, 0, i])) - d_t * h
            u[0, -1, i + 1] = u[0, -1, i] + alpha * ((2 * u[0, -2, i]) + (2 * u[1, -1, i]) - (4 * u[0, -1, i])) - d_t * h
            u[-1, -1, i + 1] = u[-1, -1, i] + alpha * ((2 * u[0, -2, i]) + (2 * u[-2, -1, i]) - (4 * u[-1, -1, i])) - d_t * h
            u[1:-1, 0, i + 1] = u[1:-1, 0, i] + alpha * ((2 * u[1:-1, 1, i]) + u[:-2, 0, i] + u[2:, 0, i] - (4 * u[1:-1, 0, i])) - d_t * h
            u[1:-1, -1, i + 1] = u[1:-1, -1, i] + alpha * ((2 * u[1:-1, -2, i]) + u[:-2, -1, i] + u[2:, -1, i] - (4 * u[1:-1, -1, i])) - d_t * h
            u[0, 1:-1, i + 1] = u[0, 1:-1, i] + alpha * ((2 * u[1, 1:-1, i]) + u[0, :-2, i] + u[0, 2:, i] - (4 * u[0, 1:-1, i])) - d_t * h
            u[-1, 1:-1, i + 1] = u[-1, 1:-1, i] + alpha * ((2 * u[-2, 1:-1, i]) + u[-1, :-2, i] + u[-1, 2:, i] - (4 * u[-1, 1:-1, i])) - d_t * h

    return u
    
"""


plt.imshow(edit_img[:,:, -1], plt.cm.gray)
plt.show()








