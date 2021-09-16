import numpy as np
import imageio
import Poisson as poi
import matplotlib.pyplot as plt
from scipy import ndimage

iter = 1

img = np.array(imageio.imread('../hdr-bilder/Balls/Balls_00128.png'))
print(img.shape)

img0 = np.sum(img.astype(float), 2) / (3 * 255.0)
plt.imshow(img0, plt.cm.gray)
plt.show()


summed_img = img[:,:, 0]/(255.0) + img[:,:, 1]/(255.0) + img[:,:,2]/(255.0)

gx, gy = np.gradient(summed_img)

rgx, rgy = np.gradient(img[:,:, 0] / (255.0))
ggx, ggy = np.gradient(img[:,:, 1] / (255.0))
bgx, bgy = np.gradient(img[:,:, 2] / (255.0))

d_len = np.sqrt((rgx**2) + (rgy**2) + (ggx**2) + (ggy**2) + (bgx**2) + (bgy**2)) / np.sqrt(3)
#d_len = (np.sqrt((rgx**2) + (rgy**2)) + np.sqrt((ggx**2) + (ggy**2)) + np.sqrt((bgx**2) + (bgy**2))) / np.sqrt(3)

g_len = (np.sqrt(gx**2 + gy**2))

gx = np.divide(gx, g_len, out=np.zeros_like(gx), where=g_len!=0.0)
gy = np.divide(gy, g_len, out=np.zeros_like(gy), where=g_len!=0.0)

gx *= d_len
gy *= d_len

gxx = ndimage.sobel(gx, axis=0)
gyy = ndimage.sobel(gy, axis=1)
h = gxx + gyy

gray = poi.poisson(img0, iter, h=h)

plt.figure()
plt.imshow(gray[:, :, -1], plt.cm.gray)
plt.show()
