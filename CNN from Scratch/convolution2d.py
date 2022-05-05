"""
THIS CLASS PERFORMS 2 DIMENSIONAL CONVOLUTION WITH STRIDE & PADDING
author: rajesh r
git: https://github.com/Rajesh-Smartino/Deep-Learning
"""

import numpy as np
import cv2
import tensorflow as tf

class Convolution():

    def __init__(self):
        pass

    def zero_padding(self, inpt, padding):

        m, n = inpt.shape

        temp = []
        for i in range(padding):
            for j in range(m + 2 * padding):
                temp.append(0)

        for i in inpt:

            for j in range(padding):
                temp.append(0)

            for p in i:
                temp.append(p)

            for k in range(padding):
                temp.append(0)

        for i in range(padding):
            for j in range(m + 2 * padding):
                temp.append(0)

        temp = np.array(temp, dtype=np.uint8)
        temp = temp.flatten()
        temp = temp.reshape(m + 2 * padding, n + 2 * padding)

        return temp

    def convolve(self, inpt, filtr, stride, padding):

        (Am, An), (fm, fn) = inpt.shape, filtr.shape
        inpt = self.zero_padding(self, inpt, padding)
        m, n = inpt.shape

        feature_map = []

        flip_leftright = np.fliplr(filtr)
        flip_updown = np.flipud(flip_leftright)
        Filtr = flip_updown

        Outm, Outn = int(((Am - fm + 2 * padding) / stride) + 1), int(((An - fn + 2 * padding) / stride) + 1)

        feature_map = []
        for i in range(0, m, stride):
            for j in range(0, n, stride):
                if inpt[i:i + fn, j:j + fm].shape == Filtr.shape:
                    feature_map.append((Filtr * inpt[i:i + fn, j:j + fm]).sum())

        feature_map = np.array(feature_map).reshape(Outm, Outn)
        return feature_map


'''
####################################################################
TO USE IT SEPERATELY, UNCOMMENT THE BELOW CODE AND RUN
####################################################################
'''
'''
obj = Convolution()

# Give the input image
inpt = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

# Give the filter
filtr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
feature_map = obj.convolve(inpt, filtr, stride = 2, padding = 1)
print(feature_map)
'''
