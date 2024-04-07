import numpy as np
import cv2

def lowpassFilter(img, kernelSize, sigma=1.0, mu=0.0):
    x, y = np.meshgrid(np.linspace(-1, 1, kernelSize), np.linspace(-1, 1, kernelSize))
    d = np.sqrt(x*x + y*y)
    kernel = np.exp(-((d - mu)**2 / (2.0 * sigma**2)))

    smoothed = cv2.filter2D(img, -1, kernel)
    return smoothed