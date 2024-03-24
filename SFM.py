import numpy as np
import cv2
import scipy
from skimage.measure import ransac
import time, os, shutil, platform
import progressbar
from tqdm import tqdm
from collections import defaultdict

class SFM:
    def __init__(self, src):
        self.get_imgs(src)
        
        # self.brisk = cv2.BRISK_create() # TODO: maybe just use BRISK - BRISK

        self.detector = cv2.BRISK_create()
        self.descriptor = cv2.SIFT_create()

    def pre_proc_img(self):
        # create bbox + get rid of radial distortion
        pass


    def get_imgs(self, src):
        self.im = []
        self.gim = []

        files = os.listdir(src)
        files.sort()
        pbar = tqdm(files)

        for f in pbar:
        # for i in progressbar.progressbar(range(len(files)), redirect_stdout=True):
            # f = files[i]
            pbar.set_description(f"{f}")
            try:
                if platform.system() == "Windows":
                    im = cv2.imread(f"{src}\{f}")
                else:
                    im = cv2.imread(f"{src}/{f}")
                    # print(f"{src}/{f}")
                gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                
                self.im.append(im)
                self.gim.append(gim)

            except:
                pbar.write(f"{f} is not an img...\n")
 
        # pass

    def ft_extract(self):
        # BRISK features
        start = time.time()

        self.kp = []
        self.desc = []
        
        print("\n\nFEATURE EXTRACTION....")
        pbar = tqdm(self.im)
        i = 1
        for im in pbar:
            pbar.set_description(f"Image {i}")
            
            kp = self.detector.detect(im)
            desc = self.descriptor.compute(im, kp)

            # kp, desc = self.brisk.detectAndCompute(im, None)

            self.kp.append(kp)
            self.desc.append(desc)
            i += 1
        
        end = time.time()
        # print(f"KEYPOINTS:\n{self.kp}\n")
        # print(f"DESCRIPTORS:\n{self.desc}")
        return self.kp, self.desc
        # print(f"ELAPSED TIME:\n{end-start}")

    def ft_match(self):
        # LO-RANSAC
        print("\nMATCHING...\n")
        self.matches = {}
        self.mask = {}
        self.M = {}
        self.src_pts = {}
        self.dst_pts = {}
        start = time.time()
        
        norm = cv2.NORM_HAMMING
        N = len(self.im)
        MIN_MATCH_COUNT = 10
        
        # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
        # TODO: change hyper params
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        self.good = defaultdict(list)
        
        pbar = tqdm(range(N-1))
        for i in pbar:
            for j in range(i+1, N):
                # print(f"{(i,j)}\n")
                pbar.set_description(f"{(i,j)}")
                self.matches[(i,j)] = flann.knnMatch(self.desc[i][1], self.desc[j][1], 2)
                
                # Lowe's
                for m, n in self.matches[(i, j)]:
                    if m.distance < 0.7 * n.distance:
                        self.good[(i,j)].append(m)

                if len(self.good[(i,j)]) > MIN_MATCH_COUNT:
                    src_pts = np.float32([ self.kp[i][m.queryIdx].pt for m in self.good[(i,j)] ]).reshape(-1,1,2)
                    dst_pts = np.float32([ self.kp[j][m.trainIdx].pt for m in self.good[(i,j)] ]).reshape(-1,1,2)

                    self.src_pts[(i,j)] = src_pts
                    self.dst_pts[(i,j)] = dst_pts

                    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_DEFAULT, 5.0)
                    # TODO: change hyper params
                    # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_FAST, 5.0)
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
                    self.M[(i,j)] = M
                    self.mask[(i,j)] = mask

                    # matchesMasks = mask.ravel().tolist()
                    
                    # h,w = self.gim[0].shape

                    # pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                else:
                    # print(f"Insufficient Matches for {(i,j)}")
                    # pbar.write(f"Insufficient Matches for {(i,j)}")
                    pbar.set_description(f"Insufficient Matches for {(i,j)}")

        end = time.time()
        elapsed = end - start # ms
        print(f"Elapsed Time: {elapsed}\n\n")


    def bundle_adjustment(self):
        pass

    def run(self):
        start = time.time()



        end = time.time()
        elapsed = end - start # ms
        pass

    '''
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
    '''



if __name__ == "__main__":
    test1 = SFM(src="test1/imgs")

    kp, desc = test1.ft_extract()