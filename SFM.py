import numpy as np
import cv2
import scipy
from skimage.measure import ransac
import time, os, shutil, platform
from tqdm import tqdm
from collections import defaultdict

class SFM:
    def __init__(self, src, K=None):
        self.K = K
        self.src = src
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
        self.itoimg = {}

        files = os.listdir(src)
        files.sort()
        pbar = tqdm(files)
        self.n = 0

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
                
                self.itoimg[self.n] = f
                self.n += 1
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
            
            # Added for visuals
            frame = cv2.drawKeypoints(im, kp, None)

            # kp, desc = self.brisk.detectAndCompute(im, None)
            # kp, desc = self.descriptor.detectAndCompute(im, None)

            self.kp.append(kp)
            self.desc.append(desc)
            i += 1
            
            cv2.imshow("f", frame)
            k = cv2.waitKey(30)
        
        end = time.time()
        # print(f"KEYPOINTS:\n{self.kp}\n")
        # print(f"DESCRIPTORS:\n{self.desc}")
        return self.kp, self.desc
        # print(f"ELAPSED TIME:\n{end-start}")

    def ft_match(self, mmc=15):
        # LO-RANSAC
        print("\nMATCHING...\n")
        self.matches = {}
        self.src_pts = {}
        self.dst_pts = {}
        
        # self.mask = {}
        # self.M = {}
        self.mask = {}
        self.F = {}
        start = time.time()
        
        norm = cv2.NORM_HAMMING
        N = self.n
        MIN_MATCH_COUNT = min(15, mmc)
        
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
                # self.matches[(i,j)] = flann.knnMatch(self.desc[i], self.desc[j], 2)
                
                # Lowe's
                for m, n in self.matches[(i, j)]:
                    if m.distance < 0.7 * n.distance:
                        self.good[(i,j)].append(m)

                if len(self.good[(i,j)]) > MIN_MATCH_COUNT:
                    src_pts = np.float32([ self.kp[i][m.queryIdx].pt for m in self.good[(i,j)] ]).reshape(-1,1,2)
                    dst_pts = np.float32([ self.kp[j][m.trainIdx].pt for m in self.good[(i,j)] ]).reshape(-1,1,2)

                    self.src_pts[(i,j)] = src_pts
                    self.dst_pts[(i,j)] = dst_pts

                    try:
                        # TODO: change hyper params
                        # F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold=4.5)
                        # F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold=3.0)
                        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, 3.0)
                        # # self.M[(i,j)] = M
                        self.F[(i,j)] = F
                        self.mask[(i,j)] = mask
                        # print(np.linalg.det(F) > 1e-7)

                        self.good[(i,j)] = np.array(self.good[(i,j)])
                        
                        if mask is None:
                            self.good[(i,j)] = []
                            continue
                        
                        # print(mask.ravel()==1)
                        
                        self.good[(i,j)] = self.good[(i,j)][mask.ravel() == 1]
                        self.good[(i,j)] = list(self.good[(i,j)])

                        if len(self.good[(i,j)]) < MIN_MATCH_COUNT:
                            self.good[(i,j)] = []
                            continue
                    except:
                        pbar.write("RANSAC FAILED, MAY CONTAIN OUTLIERS")

                else:
                    pbar.set_description(f"Insufficient Matches for {(i,j)}")
                    self.good[(i,j)] = []

        end = time.time()
        elapsed = end - start # ms
        print(f"Elapsed Time: {elapsed}\n\n")
        
        self.matches = self.good
        return self.good

    def adj_list(self, matches=None):
        if not matches:
            matches = self.good
        
        n_pairs = 0
        pairs = []
        self.adj = np.zeros((self.n, self.n))
        
        for i in range(self.n-1):
            for j in range(i, self.n):
                if len(matches[(i, j)]) > 0:
                    n_pairs += 1
                    pairs.append((i,j))
                    self.adj[i][j] = 1

        return self.adj, pairs

    
    def reconstruction(self):
        if self.K is None:
            raise AssertionError("Error: Camera matrix K not defined")
        
        # initial rec
        # best_pair = sfm_helpers.Rec.best_pair(self.adj, self.good, self.kp, self.K)
        

    def output_match(self):
        if platform.system() == "Windows":
            f = open(f"{self.src}\\match.txt", "w+")
            p = open(f"{self.src}\\pairs.txt", "w+")
        else:
            f = open(f"{self.src}/match.txt", "w+")
            p = open(f"{self.src}/pairs.txt", "w+")

        text = ""
        pairs = ""
        for i in range(self.n - 1):
            for j in range(i+1, self.n):
                if not self.good[(i, j)]:
                    continue

                im1 = self.itoimg[i]
                im2 = self.itoimg[j]

                cur = f"{im1} {im2}"
                pairs += cur + "\n"


                for m in self.good[(i, j)]:
                    q = m.queryIdx
                    t = m.trainIdx
                    cur += f"\n{q} {t}"
                cur += f"\n\n"
                text += cur

        f.write(text)
        f.close()

        p.write(pairs)
        p.close()

    def output_ft(self, i):
        im = self.itoimg[i]
        if platform.system() == "Windows":
            f = open(f"{self.src}\\{im}.txt", "w+")
        else:
            f = open(f"{self.src}/{im}.txt", "w+")
        
        n_ft = self.desc[i][1].shape[0]
        text = f"{n_ft} 128"

        for j in range(n_ft):
            x = self.desc[i][0][j].pt[0]
            y = self.desc[i][0][j].pt[1]
            scale = self.desc[i][0][j].size
            orientation = self.desc[i][0][j].angle

            D = desc[i][1][j].astype(np.int64).flatten()
            D_txt = ' '.join([f"{d}" for d in D])
            cur = f"\n{x} {y} {scale} {orientation} {D_txt}"
            text += cur

        f.write(text)
        f.close()

    def output(self):
        for i in range(self.n):
            self.output_ft(i)
        
        self.output_match()


    def bundle_adjustment(self):
        if self.K is None:
            raise AssertionError("Error: Camera matrix K not defined")
        pass

    def run(self):
        start = time.time()



        end = time.time()
        elapsed = end - start # ms
        pass


if __name__ == "__main__":
    test1 = SFM(src="test1/imgs")

    kp, desc = test1.ft_extract()