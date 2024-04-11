import numpy as np
import open3d as o3d





class Reconstruction:
    def __init__(self, poly_src):
        self.src = poly_src
        self.init_pcd = o3d.io.read_point_cloud(self.src)
        self.pcd = self.init_pcd
        self.mesh = None
        
    def reset_pcd(self, pcd):
        self.pcd = self.init_pcd

    def set_pcd(self, pcd):
        self.pcd = pcd

    def preproc(self, type="stat", down_sample=False, pcd=None, **kwargs):
        if pcd is None:
            pcd = self.init_pcd
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
            # n_neigh, std_ratio = kwargs["n_neigh"], kwargs["std_ratio"]
            try:
                n_neigh = kwargs["n_neigh"]
            except:
                print("'n_neigh' argument not provided, using default value 5")
                n_neigh = 5

            try:
                std_ratio = kwargs["std_ratio"]
            except:
                print("'std_ratio' argument not provided, using default value 0.25")
                std_ratio = 0.25
            
            _, ind = pcd.remove_statistical_outlier(n_neigh, std_ratio, False)

        elif type == "rad":
            # n_pts = 12
            # rad = .15
            # n_pts, rad = kwargs["n_pts"], kwargs["rad"]
            try:
                n_pts = kwargs["n_pts"]
            except:
                print("'n_pts' argument not provided, using default value 12")
                n_pts = 12

            try:
                rad = kwargs["rad"]
            except:
                print("'rad' argument not provided, using default value 0.15")
                rad = .15

            _, ind = pcd.remove_radius_outlier(n_pts, rad)

        else:
            print("Invalid type, no outliers removed. \nValid type values are:\n'stat' for statistical outlier removal and\n'rad' for radius outlier removal")
            ind = None
        
        pcd = pcd.select_by_index(ind)
        self.pcd = pcd
        return pcd, ind, opcd
        



    def poisson(self, depth=8, width=0, scale=1.25, l_fit=False, pcd=None, crop=True):
        # assumes normals in poly
        if pcd is None:
            pcd = self.pcd

        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=depth,
            width=width,
            scale=scale,
            linear_fit=l_fit
        )[0]

        if crop:
            bbox = pcd.get_axis_aligned_bounding_box()
            self.mesh = self.mesh.crop(bbox)

        return self.mesh

        

    def bpa(self, pcd=None, radii=None):
        # assumes normals in poly
        if pcd is None:
            pcd = self.pcd
        
        if radii is None:
            D = pcd.compute_nearest_neighbor_distance()
            avg_D = np.mean(D)
            radius = 3 * avg_D
            radii = [radius, radius * 2]
        

        self.mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd,
            o3d.utility.DoubleVector(radii)
        )

        return self.mesh


    def alpha(self):
        pass

    def create(self, mesh=None, out=None):
        if mesh is None:
            mesh = self.mesh

        if mesh is None:
            print("No mesh provided")
            return
        
        if type(mesh) is not list:
            mesh = [mesh]
        
        if out is not None and out != "":
            # create ply file
            return
        # display ply
        # o3d.visualization.draw_geometries(mesh, name="reconstructed mesh")
        o3d.visualization.draw_geometries(mesh)

        return
    
    def show_pcd(self, pcd=None, name="", show_norm=False):
        if pcd is None:
            print("Showing self.pcd point cloud (default)")
            pcd = [self.pcd]
        
        if type(pcd) is not list:
            pcd = [pcd]
        
        o3d.visualization.draw_geometries(pcd, window_name=name, point_show_normal=show_norm)