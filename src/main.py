from PyQt5 import QtWidgets, uic, QtGui

import Poisson as poi
import imageio
import numpy as np
from PyQt5 import QtWidgets
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from scipy import ndimage
import hdr_images as hdr_img


qtCreatorFile = 'main.ui'
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)


class PoissonMain(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        """
        Initierer Main klassen
        """
        QtWidgets.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self.inputBtnPoisson.clicked.connect(self.load_blur_image)
        self.outputBtnPoison.clicked.connect(self.blur_image)
        self.lagreGlatting.clicked.connect(self.save_blur)

        self.getImageBtnInpaint.clicked.connect(self.load_inpaint_image)
        self.prosImageBtnInpaint.clicked.connect(self.inpaint_image)
        self.lagreInpaint.clicked.connect(self.save_inpaint)

        self.getImageBtnFace.clicked.connect(self.load_face_image)
        self.prosImageBtnFace.clicked.connect(self.blur_Face_Color_image)
        self.lagreFace.clicked.connect(self.save_blurFace)


        self.getImageBtnMosaic.clicked.connect(self.load_mosaic_image)
        self.prosImageBtnMosaic.clicked.connect(self.mosaic_image)
        self.lagreMosaic.clicked.connect(self.save_Mosaic)

        self.getImageBtnBackClone.clicked.connect(self.load_sealess_image)
        self.getImageBtnObjectClone.clicked.connect(self.load_sealessObject_image)
        self.prosImageBtnClone.clicked.connect(self.borderless_image)
        self.lagreClone.clicked.connect(self.save_cloning)

        self.getImageBtnKontrast.clicked.connect(self.load_kontrast_image)
        self.prosImageBtnKontrast.clicked.connect(self.kontrast_image)
        self.lagreKontrast.clicked.connect(self.save_kontrast)


        self.hdr_list_model = QtGui.QStandardItemModel()
        self.image_view_list.setModel(self.hdr_list_model)
        self.add_image_hdr.clicked.connect(self.add_imgae_to_list)
        self.remove_image_hdr.clicked.connect(self.remove_image_from_list)
        self.image_view_list.selectionModel().selectionChanged.connect(self.list_item_active)
        self.hdr_process.clicked.connect(self.hdr)

        self.getImagePoiGray.clicked.connect(self.load_gray_image)
        self.poi_gray_button.clicked.connect(self.poi_gray_img)
        self.lagreGra.clicked.connect(self.save_gray)






        self.blur_image_path = None
        self.blur_img = None

        self.inpaint_image_path = None
        self.inpaint_img = None

        self.face_image_path = None
        self.face_img = None

        self.mosaic_image_path = None
        self.mosaic_img = None

        self.seamless_image_path = None
        self.seamless_img = None

        self.seamlessObject_image_path = None
        
        self.kontrast_image_path = None
        self.kontrast_img = None

        self.blur_Color_image_path = None
        self.blur_Color_img = None

        self.gray_img_path = None
        self.gray_img = None






    def get_image_path(self):
        """
        Help function that opens file explorer dialog, and gets a path to a file
        :return: Path to choosen file
        """
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName()
        return str(file_name)

    def get_rand_mode(self):
        """
        Gets which rand condition mode user has selected, either derichlet or neumann
        :return: String specifying rand condition mode
        """
        if self.dericleButtonPoisson.isChecked():
            return "dericle"
        else:
            return "neuman"

    def load_blur_image(self):
        """
        Loads Input image for blur functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.blur_image_path = path
        self.inputImageBlur.setPixmap(QtGui.QPixmap(path))

    def load_inpaint_image(self):
        """
        Loads Input image for inpainting functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.inpaint_image_path = path
        self.inputImageInpaint.setPixmap(QtGui.QPixmap(path))


    def load_face_image(self):
        """
        Loads Input image for face bluring functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.face_image_path = path
        self.inputImageFace.setPixmap(QtGui.QPixmap(path))

    def load_mosaic_image(self):
        """
        Loads Input image for mosaic functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.mosaic_image_path = path
        self.inputImageMosaic.setPixmap(QtGui.QPixmap(path))

    def load_sealess_image(self):
        """
        Loads Input image for seamless cloning (clone to image) functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.seamless_image_path = path
        self.inputBackgroundCloneImage.setPixmap(QtGui.QPixmap(path))

    def load_sealessObject_image(self):
        """
        Loads Input image for seamless cloning (clone from image) functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.seamlessObject_image_path = path
        self.inputObjectCloneImage.setPixmap(QtGui.QPixmap(path)) 

    def load_kontrast_image(self):
        """
        Loads Input image for contrast enhancing functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.kontrast_image_path = path
        self.inputImageKontrast.setPixmap(QtGui.QPixmap(path))

    def load_gray_image(self):
        """
        Loads Input image for converting to gray image functionality, and displays it as input image
        """
        path = self.get_image_path()
        self.gray_img_path = path
        self.inputGrayPoiImage.setPixmap(QtGui.QPixmap(path))









    def blur_image(self):
        """
        Implementation of image blurring functionality
        """
        #We read input image as numpy array
        img = imageio.imread(self.blur_image_path)

        #We normalize the color range from 0-255 to 0-1 space
        img = img.astype(float) / 255


        #We get rand condition type and number of iterations
        mode = self.get_rand_mode()
        if self.iterationBlur.text() != "":
            iterations = int(self.iterationBlur.text())
        else:
            iterations = 1

        #We get the lambda value which specifies max blur
        if self.blur_lambda.text() != "":
            lam = float(self.blur_lambda.text())
        else:
            lam = None

        #We specify number of channles input image has, we assume that image has either 1 or 3 channels
        channels = 0
        if img.ndim == 3:
            channels = 3
        elif img.ndim == 2:
            channels = 1
            img_ = np.zeros((img.shape[0], img.shape[1], channels))
            img_[:,:,0] = img
            img = img_

        #We loop trough all channels and solve poisson equation for each of the channels, then we combine the result
        for i in range(channels):
            blur_img = poi.poisson(img[:,:,i], iterations, rand=mode, lam=lam)
            img[:,:, i] = blur_img[:,:,-1]

        #We map the color space back from 0-1 to 0-255
        prep_img = img * 255
        prep_img = prep_img.astype(np.uint8) #We specify 1 byte per pixel value per channel
        self.blur_img = prep_img

        #We display output image
        if prep_img.shape[2] == 1:
            self.MplWidget_glatt.canvas.ax.imshow(prep_img[:,:,0].copy()/255, plt.cm.gray)
        elif prep_img.shape[2] == 3:
            self.MplWidget_glatt.canvas.ax.imshow(prep_img)

        self.MplWidget_glatt.canvas.draw()

    def inpaint_image(self):
        """
        This function performs poisson inpainting
        """

        # We read input image as numpy array
        img = imageio.imread(self.inpaint_image_path)

        # We normalize the color range from 0-255 to 0-1 space
        img = img.astype(float) / 255

        # We get rand condition type and number of iterations
        mode = self.get_rand_mode()
        if self.iterationInpaint.text() != "":
            iterations = int(self.iterationInpaint.text())
        else:
            iterations = 1

        #We create mask from specified coordinates in gui
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        mask[int(self.fromY.text()): int(self.toY.text()), int(self.fromX.text()):int(self.toX.text())] = True

        # We specify number of channles input image has, we assume that image has either 1 or 3 channels
        channels = 0
        if img.ndim == 3:
            channels = 3
        elif img.ndim == 2:
            channels = 1
            img_ = np.zeros((img.shape[0], img.shape[1], channels))
            img_[:, :, 0] = img
            img = img_

        #We loop trough all channels and solve poisson equation, then we assamble result back to a image
        for i in range(channels):
            inpaint_img = poi.poisson(img[:,:, i], iterations, rand=mode, mask=mask)
            img[:,:, i] = inpaint_img[:,:,-1]

        # We map the color space back from 0-1 to 0-255
        prep_img = img * 255
        prep_img = prep_img.astype(np.uint8) #We specify 1 byte per pixel value per channel
        self.inpaint_img = prep_img

        #We display output image
        if prep_img.shape[2] == 1:
            self.MplWidget_inpaint.canvas.ax.imshow(prep_img[:, :, 0].copy() / 255, plt.cm.gray)
        elif prep_img.shape[2] == 3:
            self.MplWidget_inpaint.canvas.ax.imshow(prep_img)

        self.MplWidget_inpaint.canvas.draw()


    def mosaic_image(self):
        """
        This function takes inn a color image, converts it to a mosiac image, and then reconstructs this mosiac back to a color image
        """
        iter = 20

        # We read input image as numpy array
        img = imageio.imread(self.mosaic_image_path)

        # We normalize the color range from 0-255 to 0-1 space
        img = img.astype(float) / 255.0

        #We convert color image to mosaic
        mosaic = np.zeros(img.shape[:2])
        mosaic[::2, ::2] = img[::2, ::2, 0]
        mosaic[1::2, ::2] = img[1::2, ::2, 1]
        mosaic[::2, 1::2] = img[::2, 1::2, 1]
        mosaic[1::2, 1::2] = img[1::2, 1::2, 2]

        #We display constructed mosaic image
        plt.imshow(mosaic)
        plt.show()

        #We extract color information form mosaic, note that not all pixels in all channels are extracted, thats why we nee poisson to reconstruct missing pixels
        reconstructed = np.zeros(img.shape)
        reconstructed[::2 ,::2, 0] = mosaic[::2, ::2]
        reconstructed[1::2, ::2, 1] = mosaic[1::2, ::2]
        reconstructed[::2, 1::2, 1] = mosaic[::2, 1::2]
        reconstructed[1::2, 1::2, 2] = mosaic[1::2, 1::2]

        #We create mask for the red channels which specifies which pixels are missing
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        mask[reconstructed[:, :, 0] == 0] = True
        red_channel = poi.poisson(reconstructed[:, :, 0], iter, mask=mask, rand='neuman') #We solve poisson equation with specified mask, we get back the reconstructed red channel

        # We create mask for the green channels which specifies which pixels are missing
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        mask[reconstructed[:, :, 1] == 0] = True
        green_channel = poi.poisson(reconstructed[:, :, 1], iter, mask=mask, rand='neuman') #We solve poisson equation with specified mask, we get back the reconstructed green channel

        # We create mask for the blue channels which specifies which pixels are missing
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        mask[reconstructed[:, :, 2] == 0] = True
        blue_channel = poi.poisson(reconstructed[:, :, 2], iter, mask=mask, rand='neuman') #We solve poisson equation with specified mask, we get back the reconstructed blue channel

        rec_img = np.zeros(reconstructed.shape)

        #We assemble all channels back together
        rec_img[:, :, 0] = red_channel[:, :, -1]
        rec_img[:, :, 1] = green_channel[:, :, -1]
        rec_img[:, :, 2] = blue_channel[:, :, -1]

        # We map color space back form 0-1 to 0-255
        rec_img = rec_img * 255
        rec_img = rec_img.astype(np.uint8) #Specify 1 byte per pixel per channel


        #Display reconstructed color image
        height, width, channel = rec_img.shape
        bytesPerLine = 3 * width
        qImg = QtGui.QImage(rec_img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)

        self.mosaic_img = rec_img


        pixmap = QtGui.QPixmap.fromImage(qImg)
        self.outputMosaic.setPixmap(pixmap)

    def borderless_image(self):
        """
        This funciton takes in 2 images, source and destination input images. It attempts to clone source image into destination image without visible transitions using poisson equation
        """
        #We read destiantion image as numpy array
        img = imageio.imread(self.seamless_image_path)

        #We normalize destination image to 0-1 color space
        img = img.astype(float) / 255.0

        #We read source image as numpy array
        img2 = imageio.imread(self.seamlessObject_image_path)

        #We normalize source image to 0-1 color space
        img2 = img2.astype(float) / 255.0

        #We read coordinates from GUI which specify where in destination image to clone source image to
        if self.fromXClone.text() != "":
            x11 = int(self.fromXClone.text())
        else:
            x11 = 1

        if self.fromYClone.text() != "":
            y11 = int(self.fromYClone.text())
        else:
            y11 = 1


        #We calcualte laplace of the source image for each channel
        laplace = np.zeros((img2.shape[0], img2.shape[1], 3))
        if img2.ndim == 3:
            for i in range(3):
                laplace[:, :, i] = ndimage.laplace(img2[:, :, i])
        else:
            for i in range(3):
                laplace[:, :, i] = ndimage.laplace(img2)

        #We create a view og the destination image of the same dimesions as source image, we use this view to solve poisson equation on
        img_to = img[y11:(y11+img2.shape[0]), x11:(x11+img2.shape[1])].copy()

        #We read number of iterations specified by user
        if self.iterationClone.text() != "":
            iterations = int(self.iterationClone.text())
        else:
            iterations = 1

        #We specify number of channels destination image has
        channels = 0
        if img.ndim == 3:
            channels = 3
        elif img.ndim == 2:
            channels = 1
            img_ = np.zeros((img.shape[0], img.shape[1], channels))
            img_[:, :, 0] = img
            img = img_

        #We loop trough all channels of destination image and assemble the result of poisson to an image
        for i in range(channels):
            trans = poi.poisson(img_to[:, :, i], iterations, h=laplace[:,:, i]) # We solve poisson equation, we specify h attribute as the laplace of source image for the current channel
            img_to[:, :, i] = trans[:, :, -1]

        #We assemble the complete image with the poisson result of the view we created
        img[y11:y11+img2.shape[0], x11:x11+img2.shape[1], :] = img_to

        #We map the image back to 0-255 color space
        prep_img = img[:,:] * 255
        prep_img = prep_img.astype(np.uint8) #1 byte per pixel per channel
        self.seamless_img = prep_img

        #We show completed image as output image
        if prep_img.shape[2] == 1:
            self.MplWidget_seam.canvas.ax.imshow(prep_img[:, :, 0].copy() / 255, plt.cm.gray)
        elif prep_img.shape[2] == 3:
            self.MplWidget_seam.canvas.ax.imshow(prep_img)

        self.MplWidget_seam.canvas.draw()



    def kontrast_image(self):
        """
        This functions attempts to enhance contrast of input image
        """
        #We read input image as numpy array
        img = imageio.imread(self.kontrast_image_path)
        #We map it to 0-1 color space
        img = img.astype(float) / 255.0

        #We get the laplace of the image for each channel
        laplace = np.zeros((img.shape[0], img.shape[1], 3))
        if img.ndim == 3:
            for i in range(3):
                laplace[:, :, i] = ndimage.laplace(img[:, :, i])
        else:
            for i in range(3):
                laplace[:, :, i] = ndimage.laplace(img)

        #We read number of iterations specified by the user
        if self.iterationKontrast.text() != "":
            iterations = int(self.iterationKontrast.text())
        else:
            iterations = 1

        #We ead contrast value specified by the user
        if self.kontrastValue.text() != "":
            kontrastValue = int(self.kontrastValue.text())
        else:
            kontrastValue = 1

        #We get the number of channels for the image
        channels = 0
        if img.ndim == 3:
            channels = 3
        elif img.ndim == 2:
            channels = 1
            img_ = np.zeros((img.shape[0], img.shape[1], channels))
            img_[:, :, 0] = img
            img = img_

        #We loop trough all channels of the image, solve poisson and assemble image
        for i in range(channels):
            ig = poi.poisson(img[:, :, i], iterations, mode='kf', laplace=laplace[:, :, i], k=kontrastValue) # We Sovle poisson by specifying laplacen of the channel and contrast values as parameters
            img[:, :, i] = ig[:, :, -1]

        #We map image back to 0-255 color space
        prep_img = img * 255
        prep_img = prep_img.astype(np.uint8)
        self.kontrast_img = prep_img

        #We display iamge to user as output image
        if prep_img.shape[2] == 1:
            self.MplWidget_con.canvas.ax.imshow(prep_img[:, :, 0].copy() / 255, plt.cm.gray)
        elif prep_img.shape[2] == 3:
            self.MplWidget_con.canvas.ax.imshow(prep_img)

        self.MplWidget_con.canvas.draw()


    def blur_Face_Color_image(self):
        """
        This function detects a face in a image and blur the face region of the iamge using OpenCV cascade classifiers
        """

        #We read image as numpy array
        img = cv2.imread(self.face_image_path)

        #We read the frontal face cascade classifier used by OpenCV
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        #We must specify how in memory the color channels gonan bu laid out (BGR because of little endian)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #We use OpenCV functionality to detect faces in an image
        faces = face_cascade.detectMultiScale(img, 1.1, 4)

        #We loop trough all detected faces in image
        for (x, y, w, h) in faces:
            #We create view of image with face and map this view to 0-1 color space
            mask = img[y:y+h, x:x+w]
            mask = mask.astype(float) / 255
            anon_img = np.zeros((mask.shape[0], mask.shape[1], mask.shape[2]))
            for i in range(img.shape[2]):
                #We solve poisson to blur the face region image for each channel, we then assemble the channels
                anon_face = poi.poisson(mask[:,:, i], 100)
                anon_img[:,:,i] = anon_face[:,:,-1]

            #We map the face region image back to 0-255 color space
            anon_img = anon_img * 255
            #We paste the blurred face region image back to the original image
            img[y:y+h, x:x+w] = anon_img[:,:,:]

        img = img.astype(np.uint8) #1 byte per pixel per channel

        #We display blurred face image as output image
        self.face_img  = img
        if img.shape[2] == 1:
            self.MplWidget_face.canvas.ax.imshow(img[:, :, 0].copy() / 255, plt.cm.gray)
        elif img.shape[2] == 3:
            self.MplWidget_face.canvas.ax.imshow(img)

        self.MplWidget_face.canvas.draw()

    

    def save_blur(self):
        if self.blur_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            if self.blur_img.shape[2] == 1:
                img = self.blur_img[:,:, 0]
            else:
                img = self.blur_img
            im = Image.fromarray(img)
            im.save(name[0])

    def save_inpaint(self):
        if self.inpaint_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            if self.inpaint_img.shape[2] == 1:
                img = self.inpaint_img[:,:, 0]
            else:
                img = self.inpaint_img
            im = Image.fromarray(img)
            im.save(name[0])

    def save_blurFace(self):
        if self.face_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            if self.face_img.shape[2] == 1:
                img = self.face_img[:,:, 0]
            else:
                img = self.face_img
            im = Image.fromarray(img)
            im.save(name[0])


    def save_Mosaic(self):
        if self.mosaic_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            im = Image.fromarray(self.mosaic_img)
            im.save(name[0])


    def save_cloning(self):
        if self.seamless_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            im = Image.fromarray(self.seamless_img)
            im.save(name[0])


    def save_kontrast(self):
        if self.kontrast_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            if self.kontrast_img.shape[2] == 1:
                img = self.kontrast_img[:,:, 0]
            else:
                img = self.kontrast_img
            im = Image.fromarray(img)
            im.save(name[0])

    def save_blurColor(self):
        if self.blurColor_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            im = Image.fromarray(self.blurColor_img)
            im.save(name[0])

    def save_gray(self):
        if self.gray_img is not None:
            name = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File', filter="*.png")
            im = Image.fromarray(self.gray_img)
            im.save(name[0])

    def add_imgae_to_list(self):
        image_path = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', 'c:\\',"Image files (*.jpg *.png)")
        self.hdr_image_preview.setPixmap(QtGui.QPixmap(image_path[0]))
        item = QtGui.QStandardItem(image_path[0])
        self.hdr_list_model.appendRow(item)


    def remove_image_from_list(self):
        selection = self.image_view_list.selectedIndexes()
        for item in selection:
            self.hdr_list_model.removeRow(item.row())

    def list_item_active(self, item):
        if len(item.indexes()) > 0:
            index = item.indexes()[0]
            path = self.hdr_list_model.itemFromIndex(index).text()
            self.hdr_image_preview.setPixmap(QtGui.QPixmap(path))


    def hdr(self):
        """
        This fucntion attempts to create a HDR imgae out of multiple images with different exposure length
        """

        #We read gamma correction value specified by the user
        str = self.gamma.text()
        if str == '':
            gamma = 1
        else:
            gamma = float(self.gamma.text())

        #We create a list of all the paths to images that user specified
        paths = [self.hdr_list_model.item(i).text() for i in range(self.hdr_list_model.rowCount())]

        #If user specified at least 1 iamge we try to run hdr processing function
        if len(paths) > 0:
            #We run the hdr processing function, the result is a constructed hdr image
            img = hdr_img.process_hdr(paths, 10)
            #We correct the constructed hdr image with our gamme value, this si because most screens dont have big enough color range to display hdr images.
            img = img**(gamma)

            #We display the HDR image as output image
            if img.shape[2] == 1:
                self.MplWidget.canvas.ax.imshow(img[:,:,0], plt.cm.gray)
            else:
                self.MplWidget.canvas.ax.imshow(img)

            self.MplWidget.canvas.draw()

    def poi_gray_img(self):
        """
        This function converts a color image to grayscale image by solving poisson equation
        """
        #We read input image as numpy array
        img = imageio.imread(self.gray_img_path)

        #We create a weighted mean sum as the base grayscale image in 0-1 color space
        img0 = np.sum(img.astype(float), 2) / (3 * 255.0)

        #We sum all the channels of input image together and map it to 0-1 color space
        summed_img = img[:, :, 0] / 255.0 + img[:, :, 1] / 255.0 + img[:, :, 2] / 255.0
        #We get the gradient of the summed channels
        gx, gy = np.gradient(summed_img)

        #We get gradient of all the channels of the input image
        rgx, rgy = np.gradient(img[:, :, 0] / (255.0))
        ggx, ggy = np.gradient(img[:, :, 1] / (255.0))
        bgx, bgy = np.gradient(img[:, :, 2] / (255.0))

        #We compute the mean length of all the gradients of all the channels of input image
        d_len = np.sqrt((rgx**2) + (rgy**2) + (ggx**2) + (ggy**2) + (bgx**2) + (bgy**2)) / np.sqrt(3)
        #We get the length of summed gradient
        g_len = (np.sqrt(gx ** 2 + gy ** 2))

        #We divide the summed gradient by its length to make it unit length
        gx = np.divide(gx, g_len, out=np.zeros_like(gx), where=g_len != 0.0)
        gy = np.divide(gy, g_len, out=np.zeros_like(gy), where=g_len != 0.0)

        #We den multiple the summed unit gradient with the cumputed mean length of all the gradients of all channels
        gx *= d_len
        gy *= d_len

        #We get the gradient of the gradient
        gxx = ndimage.sobel(gx, axis=0)
        gyy = ndimage.sobel(gy, axis=1)
        #Then we sum them, this gives as the divergence
        h = gxx + gyy

        #We read number of iterations for poisson equation specified by the user
        if self.iterationGray.text() != "":
            iterations = int(self.iterationGray.text())
        else:
            iterations = 1

        # We solve poisson equation
        gray = poi.poisson(img0, iterations, h=h)

        #We map the image back to 0-255 color space
        prep_img = gray[:,:,-1] * 255
        prep_img = prep_img.astype(np.uint8)


        #We display the output grayscale image
        self.gray_img = prep_img

        self.MplWidget_gray.canvas.ax.imshow(prep_img, plt.cm.gray)

        self.MplWidget_gray.canvas.draw()



if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = PoissonMain()
    MainWindow.show()
    sys.exit(app.exec_())
