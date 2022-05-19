import numpy as np

class convolution:

    def __init__(self):
        print("--- Convolution Layer ---")

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

        if len(filtr.shape) == 2:
            (Am, An), (fm, fn) = inpt.shape, filtr.shape
            inpt = self.zero_padding(self, inpt, padding)
            m, n = inpt.shape

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

        else:
            (D, Am, An), (d, fm, fn) = inpt.shape, filtr.shape
            padded_inpt = [] 
            for i in range(D):
                padded_inpt.append(self.zero_padding(self, inpt[i, :, :], padding))         
            inpt = np.array(padded_inpt)
            d, m, n = inpt.shape
            Outm, Outn = int(((Am - fm + 2 * padding) / stride) + 1), int(((An - fn + 2 * padding) / stride) + 1)
            feature_map = []
            for i in range(0, m, stride):
                for j in range(0, n, stride):
                    if inpt[:, i:i + fn, j:j + fm].shape == filtr.shape:
                        feature_map.append((filtr * inpt[:, i:i + fn, j:j + fm]).sum())
            
            feature_map = np.array(feature_map).reshape(Outm, Outn)
            return feature_map
        
    def inptconvolve(self, inpt, filtr, stride, padding):
        (Am, An, D), (d, fm, fn) = inpt.shape, filtr.shape

        padded_inpt = []
        for i in inpt:
            padded_inpt.append(self.zero_padding(self, i, padding))

        inpt = np.array(padded_inpt)
        m, n, d = inpt.shape

        Outm, Outn = int(((Am-fm+2*padding)/stride)+1), int(((An-fn+2*padding)/stride)+1)
        feature_map = []
        for i in range(0, m, stride):
            for j in range(0, n, stride):
                if inpt[i:i+fn, j:j+fm, :].shape == filtr.shape:
                    feature_map.append((filtr*inpt[i:i+fn, j:j+fm, :]).sum())


        feature_map = np.array(feature_map).reshape(Outm, Outn)
        return feature_map