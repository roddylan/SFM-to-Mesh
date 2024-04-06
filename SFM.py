import numpy as np
import cv2
import scipy
from skimage.measure import ransac
import time, os, shutil, platform, glob
from tqdm import tqdm
from collections import defaultdict
import database
import pycolmap, plyfile

class SFM:
    def __init__(self, src, db_path='database.db', K=None):
        self.K = K
        self.src = src
        self.get_imgs(src)
        
        self.db_path = db_path
        
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
            
            # kp, desc = self.brisk.detectAndCompute(im, None)
            # kp, desc = self.descriptor.detectAndCompute(im, None)

            self.kp.append(kp)
            self.desc.append(desc)
            i += 1
        end = time.time()
        # print(f"KEYPOINTS:\n{self.kp}\n")
        # print(f"DESCRIPTORS:\n{self.desc}")
        return self.kp, self.desc
        # print(f"ELAPSED TIME:\n{end-start}")

    def ft_match(self, mmc=12):
        # MAGSAC
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
        # MIN_MATCH_COUNT = min(15, mmc)
        # MIN_MATCH_COUNT = max(15, mmc)
        MIN_MATCH_COUNT = mmc
        
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
                        # F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold=4.5)
                        # F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold=5.0)
                        F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_MAGSAC, ransacReprojThreshold=3.5)
                        # F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.USAC_DEFAULT, ransacReprojThreshold=3.5)
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
                sz = len(matches[(i, j)])
                if sz > 0:
                    n_pairs += 1
                    pairs.append((i,j))
                    # self.adj[i][j] = 1
                    self.adj[i][j] = sz

        return self.adj, pairs
    
    def get_camera_info(self):
        for key in self.itoimg:
            camera = pycolmap.infer_camera_from_image(f'{self.src}/{self.itoimg[key]}')
            break
        return camera.model.name, camera.model.value, camera.width, camera.height, camera.params

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

    def __add_data(self):
        if os.path.exists(self.db_path):
            print("Removing existing db")
            os.remove(self.db_path)
        self.db = database.COLMAPDatabase.connect(self.db_path)
        self.db.create_tables()

        # adding camera info for pycolmap
        cam_model, cam_model_value, img_w, img_h, cam_params = self.get_camera_info()
        
        self.cam_id = self.db.add_camera(cam_model_value, img_w, img_h, cam_params)
        self.cam_obj = pycolmap.Camera(camera_id=1, model=cam_model_value, width=img_w, height=img_h, params=cam_params)
        
        # adding images
        for key in self.itoimg:
            self.db.add_image(self.itoimg[key], self.cam_id)

        self.db.commit()

        # adding keypoints and descriptors to db
        kps_per_frame = [np.array([[y.pt[0], y.pt[1], y.size, y.angle] for y in x[0]], dtype=np.float32) for x in self.desc]
        desc_per_frame = [x[1] for x in self.desc]

        for i, frame_kps in enumerate(kps_per_frame):
            self.db.add_keypoints(i+1, frame_kps)

        for i, frame_desc in enumerate(desc_per_frame):
            self.db.add_descriptors(i+1, frame_desc)

        self.db.commit()

        # adding matches and two view geo to db
        kps_2d = [np.array([[y.pt[0], y.pt[1]] for y in x[0]], dtype=np.float64) for x in self.desc]

        for i in self.good.keys():
            if isinstance(i, tuple):
                m = np.array([[x.queryIdx,x.trainIdx] for x in self.good[i]], dtype=np.uint32)
                if len(m) > 0:
                    self.db.add_matches(i[0]+1, i[1]+1, m)
                    t_geo = pycolmap.estimate_calibrated_two_view_geometry(self.cam_obj, kps_2d[i[0]], self.cam_obj, kps_2d[i[1]], m).todict()
                    self.db.add_two_view_geometry(
                        image_id1 = i[0]+1, 
                        image_id2 = i[1]+1, 
                        matches = t_geo['inlier_matches'], 
                        F = t_geo['F'], 
                        E = t_geo['E'],
                        H = t_geo['H'],
                        config = t_geo['config'])

        self.db.commit()

    def reconstruction_sparse(self, dest, obj_name='object'):
        print("\n\nSPARSE RECONSTUCTION....")
        try:
            os.mkdir(dest)
        except:
            shutil.rmtree(dest)
            os.mkdir(dest)
        self.__add_data()
        maps = pycolmap.incremental_mapping(self.db_path, self.src, dest)
        for key in maps:
            maps[key].export_PLY(f'{dest}/{obj_name}_{key}.ply')
        
        if len(maps) > 1:
            print("MERGING PLY FILES")
            files = []
            for file_name in os.listdir(dest):
                if len(file_name) < 4 or os.path.splitext(file_name)[-1].lower() != '.ply':
                    continue

                file = plyfile.PlyData.read(os.path.join(dest, file_name))
                for element in file.elements:
                    files.append(element.data)
            
            merged_file = np.concatenate(files, -1)
            merged_el = plyfile.PlyElement.describe(merged_file, "vertex")
            plyfile.PlyData([merged_el]).write(f'{dest}/{obj_name}_merged.ply')
        
    def reconstruction_dense(self, dest, obj_name='object', model_folder='/0'):
        print("\n\nDENSE RECONSTRUCTION")
        mvs = f'{dest}/mvs'
        dest += model_folder
        # try:
        #     os.mkdir(dest)
        # except:
        #     shutil.rmtree(dest)
        #     os.mkdir(dest)
        try:
            os.mkdir(mvs)
        except:
            shutil.rmtree(mvs)
            os.mkdir(mvs)
        try:
            pycolmap.undistort_images(mvs, dest, self.src)
            pycolmap.patch_match_stereo(mvs)
            pycolmap.stereo_fusion(f'{dest}/{obj_name}_dense.ply', mvs)
        except AttributeError as e:
            print("PYCOLMAP NEEDS A CUDA BUILD, PLEASE USE THE COLMAP GUI FOR DENSE RECONSTRUCTION OR BIND FROM SOURCE.")

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
    test1 = SFM(src="datasets/templeRing")

    kp, desc = test1.ft_extract()
    matches = test1.ft_match()

    test1.reconstruction_sparse('colmap_test_out')
    test1.reconstruction_dense('colmap_test_out', model_folder='/1')