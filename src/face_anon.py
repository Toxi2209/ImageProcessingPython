import numpy as np
import imageio
import Poisson as poi
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

iter=50

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img = cv2.imread('ivar_face.jpg')
print(img.shape)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)

mask = np.zeros(img.shape[:2])




for (x, y, w, h) in faces:
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
    mask[y:y+h, x:x+w] = True
    anon_face = poi.poisson(gray_img, iter, mask=mask)
    #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
plt.imshow(anon_face[:,:,-1], plt.cm.gray)
plt.show()