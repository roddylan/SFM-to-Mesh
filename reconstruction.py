import numpy as np
import open3d as o3d





class Reconstruction:
    def __init__(self, poly_src):
        self.src = poly_src
        self.init_poly = o3d.io.read_point_cloud(self.src)
        

    def preproc(self, type="stat"):
        sz = .02

        pcd = self.init_poly.voxel_down_sample(sz)

        if type == "stat":
            n_neigh = 5
            std_ratio = 0.25
            _, ind = self.init_poly.remove_statistical_outlier(n_neigh, std_ratio, False)

            
        else:
            n_pts = 12
            rad = .15
            _, ind = self.init_poly.remove_radius_outlier(n_pts, rad)

        



    def poisson(self, depth):
        pass

    def bpa(self, radius, ):
        pass

    def alpha(self):
        pass

    def create(self, out=None):
        if out != None:
            # create ply file
            return
        # display ply 
        return