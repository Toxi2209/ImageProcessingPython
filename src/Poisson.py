import numpy as np
import imageio
import matplotlib.pyplot as plt

iter = 50

def poisson(u_0, n, h=None, lam=None, rand='dericle', mask=None, mode=None, laplace=None, k=1):
    """
    Poisson equation solver implmentation using explicit method
    :param u_0: Initil condition value (initial image)
    :param n: number of iterations
    :param h: optionally specified parameter for controlling the poisson solving result (data attachment)
    :param lam: Optional parameter used with blurring to specify maximal blur
    :param rand: Specifies which rand condition type to use, supported are Derichlet and Neumann. (Derichelt is default)
    :param mask: Optionally specified parameter which allows to mask part of a image so that poisson solving wont affect masked part
    :param mode: (Optional) Specifies mode in which poisson equation is used (only supported mode is "kf" - contrast enhancing)
    :param laplace: Optional parameter to specify laplace of the image
    :param k: Optional, contrast enhancing value used with "kf" mode
    :return:
    """
    d_x = 1
    d_t = 0.25
    alpha = d_t / (d_x ** 2)

    u = np.zeros((u_0.shape[0], u_0.shape[1], n+1))

    u[:, :, 0] = u_0

    if h is None:
        h = np.zeros((u_0.shape[0], u_0.shape[1]))

    if mode == 'kf' and laplace is not None:
        h = k * laplace
    
    for i in range(n):

        if lam is not None:
            h = lam * (u[:, :, i] - u[:, :, 0])

        u[1:-1, 1:-1, i + 1] = u[1:-1, 1:-1, i] + alpha * (u[:-2, 1:-1, i] + u[2:, 1:-1, i] + u[1:-1, :-2, i] + u[1:-1, 2:, i] - (4 * u[1:-1, 1:-1, i])) - d_t * h[1:-1, 1:-1]
        if rand == 'dericle':
            u[:, 0, i + 1] = u[:, 0, i]
            u[:, -1, i + 1] = u[:, -1, i]
            u[0, :, i + 1] = u[0, :, i]
            u[-1, :, i + 1] = u[-1, :, i]
        elif rand == 'neuman':
            u[0, 0, i+1] = u[0, 0, i] + alpha * ((2 * u[0, 1, i]) +  (2 * u[1, 0, i]) - (4 * u[0, 0, i])) - d_t * h[0, 0]
            u[-1, 0, i + 1] = u[-1, 0, i] + alpha * ((2 * u[-2, 0, i]) + (2 * u[-1, 1, i]) - (4 * u[-1, 0, i])) - d_t * h[-1, 0]
            u[0, -1, i + 1] = u[0, -1, i] + alpha * ((2 * u[0, -2, i]) + (2 * u[1, -1, i]) - (4 * u[0, -1, i])) - d_t * h[0, -1]
            u[-1, -1, i + 1] = u[-1, -1, i] + alpha * ((2 * u[0, -2, i]) + (2 * u[-2, -1, i]) - (4 * u[-1, -1, i])) - d_t * h[-1, -1]
            u[1:-1, 0, i + 1] = u[1:-1, 0, i] + alpha * ((2 * u[1:-1, 1, i]) + u[:-2, 0, i] + u[2:, 0, i] - (4 * u[1:-1, 0, i])) - d_t * h[1:-1, 0]
            u[1:-1, -1, i + 1] = u[1:-1, -1, i] + alpha * ((2 * u[1:-1, -2, i]) + u[:-2, -1, i] + u[2:, -1, i] - (4 * u[1:-1, -1, i])) - d_t * h[1:-1, -1]
            u[0, 1:-1, i + 1] = u[0, 1:-1, i] + alpha * ((2 * u[1, 1:-1, i]) + u[0, :-2, i] + u[0, 2:, i] - (4 * u[0, 1:-1, i])) - d_t * h[0, 1:-1]
            u[-1, 1:-1, i + 1] = u[-1, 1:-1, i] + alpha * ((2 * u[-2, 1:-1, i]) + u[-1, :-2, i] + u[-1, 2:, i] - (4 * u[-1, 1:-1, i])) - d_t * h[-1, 1:-1]
        u[:, :, i+1] = np.clip(u[:, :, i+1], 0, 1)
        print(i)
        if mask is not None:
            f_u = u[:, :, i+1]
            f_u[np.invert(mask[:, :])] = u_0[np.invert(mask[:, :])]
            u[:, :, i+1] = f_u



    return u


if __name__ == "__main__":
    img = imageio.imread('../hdr-bilder/Adjuster/Adjuster_00032.png')
    img = np.sum(img.astype(float), 2) / (3 * 255)
    print(img.shape)

    blur_img = poisson(img, iter, rand='neuman')

    plt.imshow(blur_img[:, :, -1], plt.cm.gray)
    plt.show()




