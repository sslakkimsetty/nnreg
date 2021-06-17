import numpy as np
import matplotlib.pyplot as plt

class MSIData():
    def __init__(self, fpath_ints, fpath_pixels, fpath_features):
        self.ints = np.loadtxt(fpath_ints, dtype="str", delimiter=",")
        self.ints = self.ints[1:, 1:].astype(np.float32)

        self.pixels = np.loadtxt(fpath_pixels, dtype="str", delimiter=",")
        self.pixels = self.pixels[1:, 1:].astype(np.int32).T
        self.pixels[0] = self.pixels[0] - np.min(self.pixels[0])
        self.pixels[1] = self.pixels[1] - np.min(self.pixels[1])

        self.mz = np.loadtxt(fpath_features, dtype="str", delimiter=",")
        self.mz = self.mz[1:, 1].astype(np.float32)

        self.nFeatures = len(self.mz)
        self.areIntensitiesBuilt = False
        _ = self.showPixelLoc(quietly=True)


    def showPixelLoc(self, quietly=False):
        pixels_present = np.zeros((np.max(self.pixels[1]+1), np.max(self.pixels[0]+1))).astype(np.float32)
        pixels_present[self.pixels[1], self.pixels[0]] = 1.0

        self.nX = pixels_present.shape[0]
        self.nY = pixels_present.shape[1]

        if not quietly:
            plt.imshow(pixels_present, cmap=plt.cm.binary)
        return pixels_present


    def buildIntensityMatrix(self):
        self.intensities = np.zeros((self.nFeatures, self.nX, self.nY)).astype(np.float32)
        self.intensities[:, self.pixels[1], self.pixels[0]] = self.ints
        self.areIntensitiesBuilt = True
        return self.intensities


    def findNearestFeature(self, mz):
        idx = (np.abs(self.mz - mz)).argmin()
        return idx


    def image(self, mz, **kwargs):
        if not self.areIntensitiesBuilt:
            _ = self.buildIntensityMatrix()

        idx = self.findNearestFeature(mz)
        plt.imshow(self.intensities[idx], **kwargs)


    def exportIonImages(self, mz):
        idx = list(map(self.findNearestFeature, mz))
        idx.sort()
        return self.intensities[idx]
