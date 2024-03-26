'''
RECONSTRUCTION AND BUNDLE ADJUSTMENT CODE BASED ON rshilliday's INCREMENTAL SFM
https://github.com/rshilliday/sfm
'''

import cv2
import os, platform
import numpy as np
import random
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


class Point3D_with_views:
    def __init__(self, p3d, src_2dpt_idx):
        self.p3d = p3d
        self.src_2dpt_idx = src_2dpt_idx

class Rec:

    def best_pair(adj, matches, kp, K, thresh=.2):
        # DETERMINED BY HIGH NUMBER OF MATCHED POINTS AND HIGH APPLIED ROTATION

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
                        kpi, kpj, kpi_idx, kpj_idx = Rec.get_aligned_pts(i, j, kp, matches)
                        E, _ = cv2.findEssentialMat(kpi, kpj, K, cv2.USAC_MAGSAC, .999, 1.0)
                        points, R1, t1, mask = cv2.recoverPose(E, kpi, kpj, K)
                        rvec, _ = cv2.Rodrigues(R1) # rot mat to rot vec
                        
                        rot_angle = abs(rvec[0]) + abs(rvec[1]) + abs(rvec[2]) # sum rot angles for each dim

                        if (rot_angle > best_r or best_pair == None) and points == len(kpi): # update best img pair and check if recoverPose worked 
                            best_R = rot_angle
                            best_pair = (i, j)


        return best_pair




    def get_aligned_pts(i, j, kp, matches, mask=None):
        if mask is None:
            mask = np.ones(len(matches[(i, j)]))
        
        kpi, kpj, kpi_idx, kpj_idx = [], [], [], []

        for k in range(len(matches[(i, j)])):
            if mask[k] == 0: continue
            
            kpi.append(kp[i][matches[(i,j)][k].queryIdx].pt)
            kpi_idx.append(matches[(i,j)][k].queryIdx)

            kpj.append(kp[j][matches[(i,j)][k].trainIdx].pt)
            kpj_idx.append(matches[(i,j)][k].trainIdx)

        kpi = np.array(kpi)
        kpj = np.array(kpj)

        kpi = np.expand_dims(kpi, axis=1)
        kpj = np.expand_dims(kpj, axis=1)

        return kpi, kpj, kpi_idx, kpj_idx



    def triangulate_pts_and_reproj(R1, t1, Rr, tr, K, p3d_with_views, im_idx1, im_idx2, kpi, kpj, kpi_idx, kpj_idx, reproj=True):
        P1 = np.dot(K, np.hstack((R1, t1)))
        Pr = np.dot(K, np.hstack((Rr, tr)))

        kpi = np.squeeze(kpi)
        kpi = kpi.T
        kpi = kpi.reshape(2,-1)

        kpj = np.squeeze(kpj)
        kpj = kpj.T
        kpj = kpj.reshape(2,-1)

        p4d_hom = cv2.triangulatePoints(P1, Pr, kpi, kpj)
        p3d = cv2.convertPointsFromHomogeneous(p4d_hom.T)

        for i in range(kpi.shape[1]):
            src_2dpt_idx = {im_idx1:kpi_idx[i], im_idx2:kpj_idx[i]} # idx -> idx for 2d
            pt = Point3D_with_views(p3d[i], src_2dpt_idx)
            p3d_with_views.append(pt)

        if reproj:
            kpi = kpi.T
            kpj = kpj.T
            
            rv1, _ = cv2.Rodrigues(R1)
            rvr, _ = cv2.Rodrigues(Rr)

            projPt1, _ = cv2.projectPoints(p3d, rv1, t1, K, distCoeffs=np.array([]))
            projPtr, _ = cv2.projectPoints(p3d, rvr, tr, K, distCoeffs=np.array([]))

            del_1, del_r = [], []

            for i in range(len(projPt1)):
                del_1.append(abs( projPt1[i][0][0] - kpi[i][0] ))
                del_1.append(abs( projPt1[i][0][1] - kpi[i][1] ))
                
                del_r.append(abs( projPtr[i][0][0] - kpj[i][0] ))
                del_r.append(abs( projPtr[i][0][1] - kpj[i][1] ))

            av_err_1 = np.sum(del_1)/len(del_1)
            av_err_r = np.sum(del_r)/len(del_r)

            err = list(zip(del_1, del_r))

            return p3d_with_views, err, av_err_1, av_err_r
        return p3d_with_views


    def init_rec(kp, matches, K, pi, pj):
        kpi, kpj, kpi_idx, kpj_idx = Rec.get_aligned_pts(pi, pj, kp, matches)
        E, _ = cv2.findEssentialMat(kpi, kpj, K, cv2.USAC_MAGSAC, .999, 1.0)
        pt, R1, t1, mask = cv2.recoverPose(E, kpi, kpj, K)

        assert abs(np.linalg.det(R1)) - 1 < np.float64(1e-7)

        R0 = np.eye(3,3)
        t0 = np.zeros((3,1))

        p3d_with_views = []
        p3d_with_views = Rec.triangulate_pts_and_reproj(
            R0, t0, R1, t1, K, p3d_with_views, pi, pj, kpi, kpj, kpi_idx, kpj_idx, reproj=False
        )

        return R0, t0, R1, t1, p3d_with_views


    def get_idx_in_corr_order(idx1, idx2):
        # if idx1 < idx2: return idx1, idx2
        # else: return idx2, idx1
        return min(idx1,idx2), max(idx1,idx2)



    def adj_imgs(i, j, adj):
        if adj[i][j] == 1 or adj[j][i] ==1:
            return True
        else:
            return False



    def has_resected_pair(unres_idx, res_img, adj):
        for i in res_img:
            if adj[unres_idx][i] == 1 or adj[i][unres_idx]:
                return True
        return False
    

    def has_unresected_paur(res_idx, unres_img, adj):
        for i in unres_img:
            if adj[res_idx][i] == 1 or adj[i][res_idx] == 1:
                return True
        return False
    


    def nxt_img_pair_to_grow_recon(n, init_pair, res_img, unres_img, adj, inc=0):
        # extend rec. in both direction (resfect imgs b/w init)
        if len(unres_img) == 0: raise ValueError("Already Resected")
        
        straddle = False

        if (init_pair[1] - init_pair[0] > n/2): # init pair straddles 'end' points (ie. (0, n-1) for n imgs)
            straddle = True
            

        init_arc = init_pair[1] - init_pair[0] + 1
        
        if len(res_img) < init_arc:
            if straddle == False:
                i = res_img[-2] + 1
            else:
                i = res_img[-1] + 1
            
            while True:
                if i not in res_img:
                    prepend = True
                    unres_idx = i
                    res_idx = random.choice(res_img)
                    return res_idx, unres_idx, prepend
                i = (i + 1) % n

        extensions = len(res_img) - init_arc + inc
        if straddle == True:
            if (extensions) % 2 == 0:
                unres_idx = (init_pair[0] + int(extensions/2) + 1) % n
                res_idx = (unres_idx - 1) % n
            else:
                unres_idx = (init_pair[1] - int(extensions/2) - 1) % n
                res_idx = (unres_idx + 1) % n
        else:
            if (extensions) % 2 == 0:
                unres_idx = (init_pair[1] + int(extensions/2) + 1) % n
                res_idx = (unres_idx - 1) % n
            else:
                unres_idx = (init_pair[0] - int(extensions/2) - 1) % n
                res_idx = (unres_idx + 1) % n
        
        if adj[min(res_idx, unres_idx), max(res_idx, unres_idx)] == 0:
            possible = []
            for i in res_img:
                if adj[i, unres_idx]:
                    possible.append(i)
                if adj[unres_idx, i]:
                    possible.append(i)
            
            res_idx = random.choice(possible)



        prepend = False
        return res_idx, unres_idx, prepend
        


    def check_and_get_unres_pt(res_kpt_idx, match: cv2.DMatch, res_idx, unres_idx):
        if res_idx < unres_idx:
            if res_kpt_idx == match.queryIdx:
                unres_kpt_idx = match.trainIdx
                success = True
                return unres_kpt_idx, success
            else:
                return None, False
        elif unres_idx < res_idx:
            if res_kpt_idx == match.trainIdx:
                unres_kpt_idx = match.queryIdx
                success = True
                return unres_kpt_idx, success
            else:
                return None, False
            


    def get_corr_for_pnp(res_idx, unres_idx, pts3d, matches, kp):
        # for Perspective n Point problem
        idx1, idx2 = Rec.get_idx_in_corr_order(res_idx, unres_idx)
        triangulation_status = np.ones(len(matches[(idx1, idx2)])) # if status[x] = 1 -> matches[x] used for triangulation
        p3d_for_pnp = []
        p2d_for_pnp = []
        
        for p3d in pts3d:
            if res_idx not in p3d.src_2dpt_idx:
                continue
            res_kpt_idx = p3d.src_2dpt_idx[res_idx]
            for k in range(len(matches[(idx1, idx2)])):
                unres_kpt_idx, success = Rec.check_and_get_unres_pt(res_kpt_idx, matches[(idx1, idx2)][k], res_idx, unres_idx)
                if not success:
                    continue
                
                p3d.src_2dpt_idx[unres_idx] = unres_kpt_idx # add new 2d/3d correspondences to 3D pt obj
                p3d_for_pnp.append(p3d.p3d)
                p2d_for_pnp.append(kp[unres_idx][unres_kpt_idx].pt)
                triangulation_status[k] = 0
        
        return pts3d, p3d_for_pnp, p2d_for_pnp, triangulation_status
    


    def do_pnp(p3d_for_pnp, p2d_for_pnp, K, iter=200, reprojThresh=5):
        list_p3d_for_pnp = p3d_for_pnp
        list_p2d_for_pnp = p2d_for_pnp

        p3d_for_pnp = np.squeeze(np.array(p3d_for_pnp))
        p2d_for_pnp = np.expand_dims(
            np.squeeze(np.array(p2d_for_pnp)), axis=1
        )

        n = len(p3d_for_pnp)
        
        highest_inlier = 0

        for i in range(iter):
            pt_idx = np.random.choice(n, 6, replace=False)
            p3 = np.array(
                [p3d_for_pnp[pt_idx[i]] for i in range(len(pt_idx))]
            )
            p2 = np.array(
                [p2d_for_pnp[pt_idx[i]] for i in range(len(pt_idx))]
            )

            _, rvec, tvec = cv2.solvePnP(p3, p2, K, distCoeffs=np.array([]), flags=cv2.SOLVEPNP_ITERATIVE)
            
            R, _ = cv2.Rodrigues(rvec)

            pnp_err, projpts, avg_err, perc_inliers = Rec.test_reproj_pnp_pts(
                list_p3d_for_pnp, list_p2d_for_pnp, R, tvec, K, rep_thresh=reprojThresh
                )
            
            if highest_inlier < perc_inliers:
                highest_inlier = perc_inliers
                best_R = R
                best_tvec = tvec

        R = best_R
        
        return R, tvec


    def prep_for_reproj(img_idx, p3d_w_views, kp):
        p3d = []
        p2d = []
        p3d_idxs = []
        
        i = 0

        for pt3d in p3d_w_views:
            if img_idx in pt3d.src_2dpt_idx.keys():
                p3d_idxs.append(i)
                p3d.append(pt3d.p3d)
                kpt_idx = pt3d.src_2dpt_idx[img_idx]
                p2d.append(kp[img_idx][kpt_idx].pt)
            i += 1

        return np.array(p3d), np.array(p2d), p3d_idxs



    def calc_reproj_err(proj_pts, p2d):
        assert(len(proj_pts) == len(p2d))
        delta = []

        for i in range(len(proj_pts)):
            delta.append(
                np.abs(proj_pts[i] - p2d[i])
            )

        avg_delta = np.mean(delta)
        # avg_delta = (avg_delta[0] + )



    def test_reproj_pnp_pts(p3d_for_pnp, p2d_for_pnp, R_new, t_new, K, rep_thresh=5):
        err = []
        proj_pts = []
        inliers = []

        for i in range(len(p3d_for_pnp)):
            Xw = p3d_for_pnp[i][0]
            Xr = np.dot(R_new, Xw).reshape(3,1)
            Xc = Xr + t_new
            x = np.dot(K, Xc)
            x /= x[2]
            err.append(
                [np.float64(x[0] - p2d_for_pnp[i][0]), np.float64(x[1] - p2d_for_pnp[i][1])]
            )
            proj_pts.append(x)

            if abs(err[-1][0]) > rep_thresh or abs(err[-1][1]) > rep_thresh:
                inliers.append(0)
            else:
                inliers.append(1)
        
        a = 0

        for e in err:
            a += abs(e[0]) + abs(e[1])
        
        avg_err = a/(2*len(err))
        perc_inliers = np.mean(inliers)

        return err, proj_pts, avg_err, perc_inliers
        


class BA:
    def ba_sparsity(n_cam, n_pts, cam_ind, pt_ind):
        # 12 cam params (ext + int); 3 3D pt params
        m = cam_ind.size * 2
        n = n_cam * 12 + n_pts * 3

        # A = lil_matrix((m,n), dtype=int)
        A = lil_matrix((m, n), dtype=int)

        i = np.arange(cam_ind.size)
        
        for s in range(12):
            A[2 * i, cam_ind * 12 + s] = 1
            A[2 * i + 1, cam_ind * 12 + s] = 1

        for s in range(3):
            A[2 * i, n_cam * 12 + pt_ind * 3 + s] = 1
            A[2 * i + 1, n_cam * 12 + pt_ind * 3 + s] = 1


        return A
    
    def project(pts, cam_params, K):
        pt_proj = []

        for i in range(len(cam_params)):
            R = cam_params[i][:9].reshape(3,3)
            rvec, _ = cv2.Rodrigues(R)
            t = cam_params[i][9:]

            p = pts[i]
            p = np.expand_dims(p, axis=0)
            p, _ = cv2.projectPoints(p, rvec, t, K, distCoeffs=np.array([]))
            p = np.squeeze(np.array(p))

            pt_proj.append(p)

        return pt_proj
    
    def res(params, n_cam, n_pts, cam_ind, p_ind, p2d, K):
        cam_params = params[:n_cam * 12].reshape((n_cam, 12))
        p3d = params[n_cam * 12:].reshape((n_pts, 3))
        p_proj = BA.project(p3d[p_ind], cam_params[cam_ind], K)
        return (p_proj - p2d).ravel()


    def do_BA(p3d_with_views, R_mats, t_vecs, resec_imgs, kp, K, ftol):
        p_ind = []
        p2d = []
        cam_ind = []
        p3d = []
        cam_params = []
        BA_cam_ind = {} # true cam idx -> normalized
        cam_count = 0

        for r in resec_imgs:
            BA_cam_ind[r] = cam_count
            cam_params.append(np.hstack((R_mats[r].ravel(), t_vecs[r].ravel())))
            cam_count += 1

        for p3d_idx in range(len(p3d_with_views)):
            p3d.append(p3d_with_views[p3d_idx].p3d)
            
            for cam_idx, kpt_idx in p3d_with_views[p3d_idx].src_2dpt_idx.items():
                if cam_idx not in resec_imgs:
                    continue

                p_ind.append(p3d_idx)
                cam_ind.append(BA_cam_ind[cam_idx])
                p2d.append(kp[cam_idx][kpt_idx].pt)
        
        if len(p3d[0]) == 3:
            p3d = np.expand_dims(p3d, axis=0)

        p_ind = np.array(p_ind)
        p2d = np.array(p2d)
        cam_ind = np.array(cam_ind)
        p3d = np.squeeze(p3d)
        cam_params = np.array(cam_params)

        n_cam = cam_params.shape[0]
        n_pts = p3d.shape[0]

        x0 = np.hstack((
            cam_params.ravel(), p3d.ravel()
        ))

        A = BA.ba_sparsity(n_cam, n_pts, cam_ind, p_ind)

        res = least_squares(BA.res, x0, jac_sparsity=A, verbose=2, x_scale='jac',
                            loss='linear', ftol=ftol, xtol=1e-12, method='trf',
                            args=(n_cam, n_pts, cam_ind, p_ind, p2d, K))
        
        adj_cam_param = res.x[:n_cam * 12].reshape(n_cam, 12)
        adj_pt_3d = res.x[n_cam * 12:].reshape(n_pts, 3)

        adj_R_mats = {}
        adj_t_vecs = {}

        for true_idx, norm_idx in BA_cam_ind.items():
            adj_R_mats[true_idx] = adj_cam_param[norm_idx][:9].reshape(3,3)
            adj_t_vecs[true_idx] = adj_cam_param[norm_idx][9:].reshape(3,1)

        R_mats = adj_R_mats
        t_vecs = adj_t_vecs

        for p3d_idx in range(len(p3d_with_views)):
            p3d_with_views[p3d_idx].p3d = np.expand_dims(adj_pt_3d[p3d_idx], axis=0)

        return p3d_with_views, R_mats, t_vecs



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
