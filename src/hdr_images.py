import numpy as np
import imageio
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
import Poisson as poi



def get_weighted_value(value):
    """
    Weighting function for values
    """
    return value if value <= 128 else (255 - value)

def compute_response_curve_and_irradiance(img, ln_dt, smoothness):
    """
    The implmentation of Debevec and Malik method for estiamating response curve
    """
    img = img.astype(int)
    n = 256
    A = np.zeros((img.shape[0] * img.shape[1] * img.shape[2] + n - 1, n + img.shape[0] * img.shape[1]))
    B = np.zeros(A.shape[0])

    k = 0
    for pixel_row in range(img.shape[0]):
        for pixel_column in range(img.shape[1]):
            for image_index in range(img.shape[2]):
                pixel = img[pixel_row, pixel_column, image_index]
                pixel_index = pixel_row * img.shape[1] + pixel_column
                weighted_pixel = get_weighted_value(pixel)
                A[k, pixel] = weighted_pixel
                A[k, n + pixel_index] = -weighted_pixel
                B[k] = weighted_pixel * ln_dt[image_index]
                k += 1

    A[k, 128] = 1
    k += 1

    for i in range(n-2):
        weighted_value = get_weighted_value(i+1)
        A[k, i] = smoothness * weighted_value
        A[k, i+1] = -2 * smoothness * weighted_value
        A[k, i+2] = smoothness * weighted_value
        k += 1

    X = np.linalg.lstsq(A, B, rcond=None)

    response_curve = X[0][:n]
    irradiance = X[0][n:]

    return response_curve, irradiance


def build_response_curve(img_raw, ln_exposure_time, smooth, compression):
    #Compression value specifies how small should the estiamte imgae be relative to input image, the smaller the better performance.
    scale_percentage = compression
    width = int(img_raw[0].shape[1] * scale_percentage / 100)
    height = int(img_raw[0].shape[0] * scale_percentage / 100)
    dim = (width, height)

    img_hdr_rdy = np.zeros((height, width, len(img_raw)))

    #We resize all input images
    for i in range(len(img_raw)):
        resized = cv2.resize(img_raw[i], dim, interpolation=cv2.INTER_AREA)
        img_hdr_rdy[:, :, i] = resized

    #We return the estimated response curve
    return compute_response_curve_and_irradiance(img_hdr_rdy, ln_exposure_time, smooth)

def process_hdr(img_paths, _smoothness, compression=5):
    #img_raw = []
    R_channel = []
    G_channel = []
    B_channel = []

    channel_count = 0

    exposure_time = []
    smooth = _smoothness
    #We iterate trough each input image, we extract the color channels and expososure time from image name
    for idx, path in enumerate(img_paths):
        img = imageio.imread(path)
        #We assume that images are either RGB (3 channels) or grayscale (1 channel)
        if img.ndim > 2:
            R_channel.append(img[:,:,0])
            G_channel.append(img[:,:,1])
            B_channel.append(img[:,:,2])
            channel_count = 3
        else:
            R_channel.append(img)
            channel_count = 1
        extracted_time = int(path.split("_")[-1].split(".")[0])
        exposure_time.append(extracted_time)

    #We get the natural logarithm of the exposure times
    ln_exposure_time = np.log(np.array(exposure_time))
    #We call this funciton to estimate the response curve
    rs_curve, irr = build_response_curve(R_channel, ln_exposure_time, smooth, compression)

    #We vectorize the weighting funciton so that we can use it on whole arrays not only single values
    w_vec = np.vectorize(get_weighted_value)
    RGB = [R_channel, G_channel, B_channel]
    core_img = np.zeros((RGB[0][0].shape[0], RGB[0][0].shape[1], channel_count))
    #For each channel we estimate the HDR corrected pixel value using estimated response curve.
    for channel in range(channel_count):
        img_packed = np.zeros((RGB[channel][0].shape[0], RGB[channel][0].shape[1], len(RGB[channel])))
        for i in range(len(RGB[channel])):
            img_packed[:, :, i] = np.array(RGB[channel][i])
        img_packed = img_packed.astype(int)
        weighted_img = w_vec(img_packed.copy())
        summed_weighted_img = weighted_img.sum(2)
        summed_weighted_img[summed_weighted_img == 0] = 1
        adj_img = rs_curve[img_packed]
        for i in range(adj_img.shape[2]):
            adj_img[:, :, i] = adj_img[:, :, i] - ln_exposure_time[i]
        core_img[:,:, channel] = np.exp((weighted_img * adj_img).sum(2) / summed_weighted_img)


    return core_img



if __name__ == "__main__":

    img_paths = ["../hdr-bilder/Balls/Balls_00001.png", "../hdr-bilder/Balls/Balls_00002.png", "../hdr-bilder/Balls/Balls_00004.png",
                 "../hdr-bilder/Balls/Balls_00008.png", "../hdr-bilder/Balls/Balls_00016.png", "../hdr-bilder/Balls/Balls_00032.png",
                 "../hdr-bilder/Balls/Balls_00064.png", "../hdr-bilder/Balls/Balls_00128.png", "../hdr-bilder/Balls/Balls_00256.png",
                 "../hdr-bilder/Balls/Balls_00512.png", "../hdr-bilder/Balls/Balls_01024.png", "../hdr-bilder/Balls/Balls_02048.png"]

    img = process_hdr(img_paths, len(img_paths))
    gamma = 0.25
    img = (img ** gamma) * 1.3

    plt.imshow(img)
    plt.show()
