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

        



    def poisson(self, depth, width, scale, l_fit):
        # estimates normals
        
        return o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            self.init_poly,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=l_fit
        )[0]

        

    def bpa(self, radii: list, ):
        # assumes normals in poly
        pass

    def alpha(self):
        pass

    def create(self, out=None):
        if out != None:
            # create ply file
            return
        # display ply 
        return