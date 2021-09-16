import numpy as np
import imageio
import Poisson as poi
import matplotlib.pyplot as plt
from scipy import ndimage

iter = 50

img = imageio.imread('../hdr-bilder/Bonita/Bonita_00512.png')
print(img.shape)
img = np.sum(img.astype(float), 2) / (3 * 255)

img2 = imageio.imread('make.png')
print(img2.shape)
img2 = np.sum(img2.astype(float), 2) / (3 * 255)

x1 = 100
y1 = 100
x2 = 150
y2 = 150

x11 = 200
y11 = 200

laplace = ndimage.laplace(img2)

laplace_o1 = laplace[y1:y2, x1:x2]
img_to = img[y11:(y11+img2.shape[0]), x11:(x11+img2.shape[1])]

trans = poi.poisson(img_to, iter, h=laplace)

img[y11:y11+img2.shape[0], x11:x11+img2.shape[1]] = trans[:,:,-1]

plt.imshow(img[:,:], plt.cm.gray)
plt.show()