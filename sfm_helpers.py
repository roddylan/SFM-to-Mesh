import cv2
import os, platform
import numpy as np

class Rec:
    @staticmethod
    def best_pair(adj, matches, kp, K, thresh=.2):

        n_matches = []

        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i,j] == 1:
                    n_matches.append(len(matches[(i,j)]))

        n_matches.sort(reverse=True)
        min_idx = int(len(n_matches) * thresh)
        min_matches = n_matches[min_idx]

        best_r = 0
        best_pair = None

        for i in range(adj.shape[0]):
            for j in range(adj.shape[1]):
                if adj[i,j] == 1:
                    if len(matches[(i,j)]) > min_matches:
                        kpi, kpj, kpi_idx, kpj_idx = Rec.get_aligned_pts()


    @staticmethod
    def get_aligned_pts(i, j, kp, matches, mask=None):
        if mask is None:
            mask = np.ones(len(matches[(i, j)]))

        
        




class Helper:
    
    @staticmethod
    def load_img(src):
        files = os.listdir(src)
        files.sort()
        imgs = []
        gimgs = []
        for f in files:
            if platform.system() == "Windows":
                path = f"{src}\\{f}"
            else:
                path = f"{src}/{f}"
            try:
                im = cv2.imread(path)
                gim = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

                imgs.append(im)
                gimgs.append(gim)
            except:
                print(f"{path} not an img")

        return imgs, gimgs

    @staticmethod
    def get_intrinsic(cb_sz: tuple, res: tuple, imgs, gimgs):
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, .001)

        objp = np.zeros((
            cb_sz[0] * cb_sz[1], 3
        ), np.float32)
        objp[:,:2] = np.mgrid[0:cb_sz[0], 0:cb_sz[1]].T.reshape(-1,2)

        objPoints = [] # 3d real
        imgPoints = [] # 2d img plane

        for im, g in zip(imgs, gimgs):
            ret, corners = cv2.findChessboardCorners(im, cb_sz, None)

            if ret == True:
                objPoints.append(objp)
                c2 = cv2.cornerSubPix(g, corners, (11, 11), (-1, -1), criteria)
                imgPoints.append(corners)

                cv2.drawChessboardCorners(im, cb_sz, c2, ret)
                cv2.imshow('im', im)

                cv2.waitKey(1000)

        cv2.destroyAllWindows()

        ret, cmat, dist, rvecs, tvecs = cv2.calibrateCamera(objPoints, imgPoints, res, None, None)

        return ret, cmat
