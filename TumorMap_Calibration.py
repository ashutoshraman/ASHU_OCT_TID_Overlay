

'''
1. initial calibration
2. sim platform
3. exp platform

Pipeline:
1. p(x,y) -> initial point
2. ROI scanning region
3. threshold -> Local ROI.

Steps:
S1:
    1. define a map in Sim (label the fiducial positions) -> actual units
    2. define the stage local frame
    3. define initial fiducials
S2:
    1. register the two coordinate frames
    2. define a virtual scan
    3. estimate the robot motion stages

refernce:
1. 3D rigid transformation: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py

'''

# import Lib_DataProcess
# import Lib_EngineSim
# import Lib_TumorMap2D
import numpy as np
import matplotlib.pyplot as plt
import scipy

class TumorMap(): #tumormap

    def __init__(self):

        # define the class
        # self.Lib_DataProcess = Lib_DataProcess.TumorID()
        # self.Lib_EngineSim = Lib_EngineSim.eng()
        # self.Lib_TumorMap2D = Lib_TumorMap2D.EngineRobot()

        # define the ranges
        self.Len_w = 30         # 30 mm
        self.Len_h = 30         # 30 mm
        self.ROI_scan = 5       # 5 * 5 mm^2 scanning range
        self.ROI_step = 1       # 1.0 mm scanning steps

        # global map -> stage
        Nx_new = self.Len_w + 1
        Ny_new = self.Len_h + 1
        x_min = y_min = 0.0
        x_max = self.Len_w
        y_max = self.Len_h
        rx, ry = np.arange(x_min, x_max, (x_max - x_min) / Nx_new), np.arange(y_min, y_max, (y_max - y_min) / Ny_new)
        grid_x, grid_y = np.meshgrid(rx, rx)
        self.data_grid = np.zeros((len(grid_x.ravel()), 2))
        self.data_grid[:, 0] = grid_x.ravel()
        self.data_grid[:, 1] = grid_y.ravel()

        # local map -> scanning region
        Len_ref = 10.0
        self.Len_w_Local = self.Len_h_Local = 20
        Nx_new = self.Len_w_Local + 1
        Ny_new = self.Len_h_Local + 1
        x_min = y_min = 0.0
        x_max = Len_ref
        y_max = Len_ref
        rx, ry = np.arange(x_min, x_max, (x_max - x_min) / Nx_new), np.arange(y_min, y_max, (y_max - y_min) / Ny_new)
        grid_x, grid_y = np.meshgrid(rx, rx)
        self.data_grid_local = np.zeros((len(grid_x.ravel()), 2))
        self.data_grid_local[:, 0] = grid_x.ravel()
        self.data_grid_local[:, 1] = grid_y.ravel()

        # define fiducial positions
        self.Len_fid = Len_ref / 2.0 * np.sqrt(2.0)
        self.fid_cen = [ self.Len_w / 2.0, self.Len_h / 2.0 ]
        self.fid_1 = [ self.fid_cen[0] - self.Len_fid, self.fid_cen[1] ]
        self.fid_2 = [ self.fid_cen[0], self.fid_cen[1] + self.Len_fid ]
        self.fid_3 = [ self.fid_cen[0] + self.Len_fid, self.fid_cen[1] ]
        self.fid_4 = [self.fid_cen[0], self.fid_cen[1] - self.Len_fid]

    def Tform_Vector(self, pts_1, pts_2, pts_3):

        # vector format
        # scale_norvec = 5.0
        fid_scan_1 = np.asarray(self.fid_1)
        fid_scan_2 = np.asarray(self.fid_2)
        fid_scan_3 = np.asarray(self.fid_4)
        fid_scan_v1 = fid_scan_2 - fid_scan_1
        fid_scan_v1 = fid_scan_v1 / np.linalg.norm(fid_scan_v1)
        fid_scan_v2 = fid_scan_3 - fid_scan_1
        fid_scan_v2 = fid_scan_v2 / np.linalg.norm(fid_scan_v2)

        # # test-1
        # Len_exp = 5.0
        # pts_scan_1 = np.asarray([Len_exp, Len_exp])
        # pts_stage_1 = fid_scan_1 + fid_scan_v1 * pts_scan_1[0] + fid_scan_v2 * pts_scan_1[1]

        # report the results
        pts_org = fid_scan_1
        vec_1 = fid_scan_v1
        vec_2 = fid_scan_v2

        return pts_org, vec_1, vec_2

    def Tform_Matice(self):

        # get the vectors
        pts_org, vec_1, vec_2 = self.Tform_Vector(self.fid_1, self.fid_2, self.fid_4)

        # matrix format + find the affine transformation
        scale_axis = 5.0
        fid_stage_1 = pts_org
        fid_stage_2 = pts_org + vec_1 * scale_axis
        fid_stage_3 = pts_org + vec_2 * scale_axis

        # registration of the function programs
        fid_local_1 = np.asarray([0.0, 0.0])
        fid_local_2 = np.asarray([0.0, 1.0 * scale_axis])
        fid_local_3 = np.asarray([1.0 * scale_axis, 0.0])
        data_rigid_A = np.vstack([fid_local_1, fid_local_2, fid_local_3])
        data_rigid_B = np.vstack([fid_stage_1, fid_stage_2, fid_stage_3])
        R_scan2stage, t_scan2stage = rigid_transform_3D(np.transpose(data_rigid_A), np.transpose(data_rigid_B))
        data_rigid_A_proj = np.matmul(R_scan2stage, np.transpose(data_rigid_A)) + t_scan2stage

        # test the results
        pts_scan_2 = np.asarray([5.0, 5.0])
        pts_stage_2_test = np.matmul(R_scan2stage, np.transpose(pts_scan_2)) + np.transpose(t_scan2stage)

        # vis the program
        Len_text = 0.5
        Len_fontsize = 10

        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)

        # Stage frame
        self.ax1.scatter(self.data_grid[:, 0], self.data_grid[:, 1])
        self.ax1.scatter(fid_stage_1[0], fid_stage_1[1], c='r')
        self.ax1.text(fid_stage_1[0] + Len_text, fid_stage_1[1] + Len_text, "fid-stage-1", fontsize = Len_fontsize)
        self.ax1.scatter(fid_stage_2[0], fid_stage_2[1], c='r')
        self.ax1.text(fid_stage_2[0] + Len_text, fid_stage_2[1] + Len_text, "fid-stage-2", fontsize = Len_fontsize)
        self.ax1.scatter(fid_stage_3[0], fid_stage_3[1], c='r')
        self.ax1.text(fid_stage_3[0] + Len_text, fid_stage_3[1] + Len_text, "fid-stage-3", fontsize = Len_fontsize)
        self.ax1.scatter(self.fid_1[0], self.fid_1[1], c = 'b', marker = "*")
        self.ax1.scatter(self.fid_2[0], self.fid_2[1], c = 'b', marker = "*")
        self.ax1.scatter(self.fid_3[0], self.fid_3[1], c = 'b', marker = "*")
        self.ax1.scatter(self.fid_4[0], self.fid_4[1], c = 'b', marker = "*")
        self.ax1.set_title('Frame-stage')
        self.ax1.set_aspect('equal', 'box')
        self.ax1.set_xlabel("unit: mm")
        self.ax1.set_ylabel("unit: mm")

        # Local frame
        self.ax2.scatter(self.data_grid_local[:,0], self.data_grid_local[:,1])
        # self.ax2.scatter(pts_traj[:,0], pts_traj[:,1], c = 'g')
        self.ax2.set_title('Frame-local')
        self.ax2.set_aspect('equal', 'box')
        self.ax2.set_xlabel("unit: mm")
        self.ax2.set_ylabel("unit: mm")

        # plt.axis('equal')
        # plt.show()

        return R_scan2stage, t_scan2stage

    def calibration(self):

        # step-1: define the initial guess (manually move the stage)
        self.Len_fid = 5.0 * np.sqrt(2.0)
        self.fid_cen = [self.Len_w / 2.0, self.Len_h / 2.0]
        self.fid_1 = [self.fid_cen[0] - self.Len_fid, self.fid_cen[1]]
        self.fid_2 = [self.fid_cen[0], self.fid_cen[1] + self.Len_fid]
        self.fid_3 = [self.fid_cen[0] + self.Len_fid, self.fid_cen[1]]
        self.fid_4 = [self.fid_cen[0], self.fid_cen[1] - self.Len_fid]

        # # step-2: scanning the regions + estimate the guess region
        # x_fid_1, y_fid_1 = self.Lib_DataProcess.calibration(self.fid_1[0], self.fid_1[1], r = 5.0, n = 10)
        # x_fid_2, y_fid_2 = self.Lib_DataProcess.calibration(self.fid_2[0], self.fid_2[1], r = 5.0, n = 10)
        # x_fid_3, y_fid_3 = self.Lib_DataProcess.calibration(self.fid_3[0], self.fid_3[1], r = 5.0, n = 10)
        # x_fid_4, y_fid_4 = self.Lib_DataProcess.calibration(self.fid_4[0], self.fid_4[1], r = 5.0, n = 10)
        #
        # # step-3: update the global fiducial positions
        # self.fid_1 = [x_fid_1, y_fid_1]
        # self.fid_2 = [x_fid_2, y_fid_2]
        # self.fid_3 = [x_fid_3, y_fid_3]
        # self.fid_4 = [x_fid_4, y_fid_4]

        # step-4: registration
        R_scan2stage, t_scan2stage = self.Tform_Matice()

        # testing
        # local trajectory
        pts_Ltraj_x = np.linspace(2.0, 8.0, num = 20).ravel()
        pts_Ltraj_y = np.sin(pts_Ltraj_x).ravel()
        pts_Ltraj_y = pts_Ltraj_y + 2.0
        pts_Ltraj = np.transpose(np.vstack([pts_Ltraj_x, pts_Ltraj_y]))
        print("pts_Ltraj = ", pts_Ltraj.shape)

        # global trajectory
        pts_Gtraj = np.transpose(np.matmul(R_scan2stage, np.transpose(pts_Ltraj)) + t_scan2stage)
        print("pts_Gtraj = ", pts_Gtraj.shape)

        # vis the programs
        # self.fig = plt.figure()
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)

        # Stage frame
        self.ax3.scatter(self.data_grid[:, 0], self.data_grid[:, 1])
        self.ax3.scatter(pts_Gtraj[:,0], pts_Gtraj[:,1], c = 'r', marker = "*")
        self.ax3.scatter(self.fid_1[0], self.fid_1[1], c = 'b', marker = "*")
        self.ax3.scatter(self.fid_2[0], self.fid_2[1], c = 'b', marker = "*")
        self.ax3.scatter(self.fid_3[0], self.fid_3[1], c = 'b', marker = "*")
        self.ax3.scatter(self.fid_4[0], self.fid_4[1], c = 'b', marker = "*")
        self.ax3.set_title('Frame-stage')
        self.ax3.set_aspect('equal', 'box')
        self.ax3.set_xlabel("unit: mm")
        self.ax3.set_ylabel("unit: mm")

        # Local frame
        self.ax4.scatter(self.data_grid_local[:,0], self.data_grid_local[:,1])
        self.ax4.scatter(pts_Ltraj[:,0], pts_Ltraj[:,1], c = 'r', marker = "*")
        self.ax4.set_title('Frame-local')
        self.ax4.set_aspect('equal', 'box')
        self.ax4.set_xlabel("unit: mm")
        self.ax4.set_ylabel("unit: mm")

        plt.axis('equal')
        plt.show()

def rigid_transform_3D(A, B):

    '''
    refernce: https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
    '''

    assert A.shape == B.shape

    # num_rows, num_cols = A.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    #
    # num_rows, num_cols = B.shape
    # if num_rows != 3:
    #     raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t

if __name__ == "__main__":

    obj = TumorMap()
    obj.calibration()


