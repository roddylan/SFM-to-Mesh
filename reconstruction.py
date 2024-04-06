import numpy as np
import open3d as o3d





class Reconstruction:
    def __init__(self, poly_src):
        self.src = poly_src
        self.init_poly = o3d.io.read_point_cloud(self.src)
        

    def preproc(self, type="stat", down_sample=False, pcd=None, **kwargs):
        if pcd is None:
            pcd = self.init_poly
        opcd = pcd
        
        if down_sample:
            # sz = .02
            try:
                sz = kwargs["sz"]
            except:
                print("Size argument for downsampling, 'sz', not given, using default 0.02 (0.05 recommended if you want more sparse result)")
                sz = .02
            pcd = pcd.voxel_down_sample(sz)
        
        
        if type == "stat":
            # n_neigh = 5
            # std_ratio = 0.25
            n_neigh, std_ratio = kwargs["n_neigh"], kwargs["std_ratio"]
            _, ind = pcd.remove_statistical_outlier(n_neigh, std_ratio, False)

        elif type == "rad":
            # n_pts = 12
            # rad = .15
            n_pts, rad = kwargs["n_pts"], kwargs["rad"]
            _, ind = pcd.remove_radius_outlier(n_pts, rad)

        else:
            print("Invalid type, no outliers removed. \nValid type values are:\n'stat' for statistical outlier removal and\n'rad' for radius outlier removal")
            ind = None
        
        pcd = pcd.select_by_index(ind)
        return pcd, ind, opcd
        



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