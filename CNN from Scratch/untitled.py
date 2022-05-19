"""
THIS CLASS BUILDS THE CONVOLUTION LAYERS
author: rajesh r
git: https://github.com/Rajesh-Smartino/Deep-Learning
"""

import numpy as np
from convolution2d import Convolution as cv


class layer():

    def __init__(self):
        pass

    def kaiming(self, n, size):
        mean, sd = 0, np.sqrt(2 / n)
        weights = np.random.normal(mean, sd, size=size)
        return weights

    def ReLUActivation(self, x):
        out = []
        for i in x:
            for j in i:
                if j >= 0:
                    out.append(j)
                else:
                    out.append(0)
        return np.array(out).reshape(x.shape)

    def ConVlayer(self, image, K, size, n):
        # n = 1
        output = []
        ReLUout = []
        for i in range(K):
            filtr = self.kaiming(n, size)
            res = cv.convolve(cv, inpt=image, filtr=filtr, stride=1, padding=0)
            output.append(res)
            ReLUout.append(self.ReLUActivation(res))

        self.output = np.array(output)
        self.ReLUout = np.array(ReLUout)

    def ConVDepthlayer(self, images, K, size, n):
        output = []
        ReLUout = []
        for i in range(K):
            filtr = self.kaiming(n, size)
            res = cv.convwithDepth(cv, inpt=images, filtr=filtr, stride=1, padding=0)
            output.append(res)
            ReLUout.append(self.ReLUActivation(res))

        self.output = np.array(output)
        self.ReLUout = np.array(ReLUout)

    def poolingLayer(self, featureMaps, size, stride):
        m, n = featureMaps[0].shape
        fm, fn = size
        result = []

        poolout = []
        for featureMap in featureMaps:
            for i in range(0, m, stride):
                for j in range(0, n, stride):
                    if featureMap[i:i + fm, j:j + fn].shape == size:
                        result.append(max(featureMap[i:i + fm, j:j + fn].flatten()))

            Outm, Outn = int(((m - fm) / stride) + 1), int(((n - fn) / stride) + 1)
            poolout.append(np.array(result).reshape((Outm, Outn)))
            result = []

        self.poolout = np.array(poolout)

