import numpy as np
import cv2
import scipy
from skimage.measure import ransac  
import time, os, shutil

class SFM:
    def __init__(self, src):
        self.get_imgs(src)
        self.detector = cv2.BRISK_create()
        self.descriptor = cv2.xfeatures2d.SIFT_create()
        # pass

    def pre_proc_img(self):
        # bbox + radial distortion
        pass


    def get_imgs(self, src):
        self.im = []
        self.gim = []

        files = os.listdir(src)
        files.sort()

        for f in files:
            im = cv2.imread(files)
            self.im.append(im)
            gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            self.gim.append(gim)
 
        # pass

    def ft_extract(self):
        # BRISK features
        self.kp = []
        
        pass


    def ft_match(self):
        # RANSAC
        start = time.time()



        end = time.time()
        elapsed = end - start # ms
        pass


    def bundle_adjustment(self):
        pass

    def run(self):
        start = time.time()



        end = time.time()
        elapsed = end - start # ms
        pass


    def _fmatch(self, kp1, kp2, matches, ratio = 0.75):
        mkp1, mkp2, good= [], [], []
        for m in matches:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                m = m[0]
                mkp1.append( kp1[m.queryIdx] )
                mkp2.append( kp2[m.trainIdx] )
                good.append(m)
        p1 = np.float32([kp.pt for kp in mkp1])
        p2 = np.float32([kp.pt for kp in mkp2])
        kp_pairs = zip(mkp1, mkp2)
        return p1, p2, kp_pairs, good
    

    def _smatch(self):
        pass




if __name__ == "__main__":
    test1 = SFM(src="test1")