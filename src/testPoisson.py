import unittest
import numpy as np
import imageio
import matplotlib.pyplot as plt
import Poisson as poi


class test_Poisson(unittest.TestCase):

    def test_dericle(self):
        img = imageio.imread('../hdr-bilder/Adjuster/Adjuster_00032.png')
        img = np.sum(img.astype(float), 2) / (3 * 255)
        self.assertEqual((poi.poisson(img, 3, rand='dericle')[0,0,-1]),img[0,0])
        self.assertEqual((poi.poisson(img, 3, rand='dericle')[-1,-1,-1]),img[-1,-1])


    def test_neumann(self):
        img = imageio.imread('../hdr-bilder/Adjuster/Adjuster_00032.png')
        img = np.sum(img.astype(float), 2) / (3 * 255)
        self.assertNotEqual((poi.poisson(img, 3, rand='neuman')[0,0,-1]),img[0,0])
        self.assertNotEqual((poi.poisson(img, 3, rand='neuman')[-1,-1,-1]),img[-1,-1])    


    def test_inpaint(self):
        img = imageio.imread('../hdr-bilder/Adjuster/Adjuster_00032.png')
        img = np.sum(img.astype(float), 2) / (3 * 255)
        mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        mask[50:100, 50:100] = True
        inpaint_img = poi.poisson(img[:,:], 5, rand="dericle", mask=mask)
        self.assertEqual(inpaint_img[1,1,-1],img[1,1])
        self.assertEqual(inpaint_img[49,49,-1],img[49,49])
        self.assertNotEqual(inpaint_img[55,55,-1],img[55,55])
        self.assertNotEqual(inpaint_img[95,95,-1],img[95,95])
        self.assertEqual(inpaint_img[101,101,-1],img[101,101])

if __name__ == '__main__':
    unittest.main()