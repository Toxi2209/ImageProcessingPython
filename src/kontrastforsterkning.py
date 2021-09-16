import numpy as np
import imageio
import Poisson as poi
import matplotlib.pyplot as plt
from scipy import ndimage



img = imageio.imread('../hdr-bilder/Bonita/Bonita_00512.png')
img = np.sum(img.astype(float), 2) / (3 * 255)
print(img.shape)

sx = ndimage.sobel(img, axis=0, mode='constant')
sy = ndimage.sobel(img, axis=1, mode='constant')

sxx = ndimage.sobel(sx, axis=0, mode='constant')
syy = ndimage.sobel(sy, axis=1, mode='constant')

laplace = sxx + syy

laplace3 = ndimage.laplace(img)

plt.imshow(laplace3, plt.cm.gray)
plt.show()

laplace2 = np.ones((img.shape[0], img.shape[1]))
laplace2[1:-1, 1:-1] = img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:] - (4 * img[1:-1, 1:-1])

dudx = img[2:, 1:-1] - img[1:-1, 1:-1]
dudy = img[1:-1, 2:] - img[1:-1, 1:-1]

laplace4 = np.zeros((img.shape[0], img.shape[1]))
laplace4[2:-2, 2:-2] = dudx[1:-1, 1:-1] - dudx[:-2, 1:-1] + dudy[1:-1, 1:-1] - dudy[1:-1, :-2]

"""
plt.imshow(laplace, plt.cm.gray)
plt.figure()
plt.imshow(laplace2, plt.cm.gray)
plt.show()
"""

ig = poi.poisson(img, 120, mode='kf', laplace=laplace3, k=2)


plt.imshow(ig[:, :, -1], plt.cm.gray)
plt.show()


