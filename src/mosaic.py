import numpy as np
import imageio
import Poisson as poi
import matplotlib.pyplot as plt
from scipy import ndimage

iter = 20

img = imageio.imread('../hdr-bilder/Bonita/Bonita_00512.png')
print(img.shape)

mosaic = np.zeros(img.shape[:2])
mosaic[::2, ::2] = img[::2, ::2, 0]
mosaic[1::2, ::2] = img[1::2, ::2, 1]
mosaic[::2, 1::2] = img[::2, 1::2, 1]
mosaic[1::2, 1::2] = img[1::2, 1::2, 2]


reconstructed = np.zeros(img.shape)
reconstructed[::2 ,::2, 0] = mosaic[::2, ::2]
reconstructed[1::2, ::2, 1] = mosaic[1::2, ::2]
reconstructed[::2, 1::2, 1] = mosaic[::2, 1::2]
reconstructed[1::2, 1::2, 2] = mosaic[1::2, 1::2]

mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
mask[reconstructed[:, :, 0] == 0] = True

red_channel = poi.poisson(reconstructed[:, :, 0], iter, mask=mask, rand='neuman')

mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
mask[reconstructed[:, :, 1] == 0] = True

green_channel = poi.poisson(reconstructed[:, :, 1], iter, mask=mask, rand='neuman')

mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
mask[reconstructed[:, :, 2] == 0] = True

blue_channel = poi.poisson(reconstructed[:, :, 2], iter, mask=mask, rand='neuman')


rec_img = np.zeros(img.shape)
rec_img[:, :, 0] = red_channel[:, :, -1]
rec_img[:, :, 1] = green_channel[:, :, -1]
rec_img[:, :, 2] = blue_channel[:, :, -1]

rec_img = rec_img.astype(np.uint8)

plt.imshow(rec_img)
plt.show()
