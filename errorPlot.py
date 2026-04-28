import matplotlib.pyplot as plt
import numpy as np
import os
import math
def align_3d_points_with_svd(gt_points, est_points, find_scale=True):
    assert len(gt_points) == len(est_points), "The number of points must be the same"
    is_ok = False

    # Next, align the two trajectories on the basis of their associations
    gt = np.array(gt_points).T  # 3xN
    est = np.array(est_points).T  # 3xN

    mean_gt = np.mean(gt, axis=1)
    mean_est = np.mean(est, axis=1)

    gt -= mean_gt[:, None]
    est -= mean_est[:, None]

    cov = np.dot(gt, est.T)
    if find_scale:
        # apply Kabsch–Umeyama algorithm
        cov /= gt.shape[0]
        variance_gt = np.mean(np.linalg.norm(gt, axis=1) ** 2)

    try:
        U, D, Vt = np.linalg.svd(cov)
    except:
        print("[align_3d_points_with_svd] SVD failed!!!\n")
        return np.eye(4), is_ok

    c = 1
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    if find_scale:
        # apply Kabsch–Umeyama algorithm
        c = variance_gt / np.trace(np.diag(D) @ S)

    rot_gt_est = np.dot(U, np.dot(S, Vt))
    trans = mean_gt - c * np.dot(rot_gt_est, mean_est)

    T_gt_est = np.eye(4)
    T_gt_est[:3, :3] = c * rot_gt_est
    T_gt_est[:3, 3] = trans

    T_est_gt = np.eye(4)  # Identity matrix initialization
    R_gt_est = T_gt_est[:3, :3]
    t_gt_est = T_gt_est[:3, 3]
    if find_scale:
        # Compute scale as the average norm of the rows of the rotation matrix
        s = c  # np.mean([np.linalg.norm(R_gt_est[i, :]) for i in range(3)])
        R = rot_gt_est  # R_gt_est / s
        sR_inv = (1.0 / s) * R.T
        T_est_gt[:3, :3] = sR_inv
        T_est_gt[:3, 3] = -sR_inv @ t_gt_est.ravel()
    else:
        T_est_gt[:3, :3] = R_gt_est.T
        T_est_gt[:3, 3] = -R_gt_est.T @ t_gt_est.ravel()

    is_ok = True
    return T_gt_est, T_est_gt, is_ok

def associate(first_list, second_list, offset=0, max_difference=1/41,startTime=0.0):
        """
        Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim
        to find the closest match for every input tuple.

        Input:
        first_list -- first list of (stamp,data) tuples
        second_list -- second list of (stamp,data) tuples
        offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
        max_difference -- search radius for candidate generation

        Output:
        matches -- map: index_stamp_first -> (index_stamp_second, diff_stamps, first_timestamp, second_timestamp)
        """
        matches = {}
        first_flag = [False] * len(first_list)
        second_flag = [False] * len(second_list)
        # extract timestamps
        t1 = np.ascontiguousarray([float(data1) for data1 in first_list])
        t2 = np.ascontiguousarray([(float(data2) + offset) for data2 in second_list])
        for i, t in enumerate(t1):
            j = np.argmin(np.abs(t2 - t))
            if abs(t2[j] - t) < max_difference and t>t1[0]+startTime:
                first_flag[i] = True
                second_flag[j] = True
                #matches[int(i)] = (int(j), abs(t2[j] - t), t, t2[j])
                matches[int(i)] = (int(i),int(j),t1[i])
        missing_associations = [(i, a) for i, a in enumerate(first_list) if first_flag[i] is False]
        num_missing_associations = len(missing_associations)
        print(f"[associate] Number of matches: {len(matches)}, number of missing associations: {num_missing_associations}")
        return matches

def yaw_from_cv_quaternion(qx, qy, qz, qw):

    # Correct CV->NED quaternion:
    cx, cy, cz, cw = 0.5, 0.5, 0.5, 0.5

    # Quaternion multiplication: q_ned = q_cv_to_ned ⊗ q_cv
    rx = cw*qx + cx*qw + cy*qz - cz*qy
    ry = cw*qy - cx*qz + cy*qw + cz*qx
    rz = cw*qz + cx*qy - cy*qx + cz*qw
    rw = cw*qw - cx*qx - cy*qy - cz*qz

    # Yaw extraction for NED (clockwise-positive!)
    siny = 2 * (rw*rz + rx*ry)
    cosy = 1 - 2 * (ry*ry + rz*rz)
    return math.atan2(siny, cosy)

def quat_to_R(qx, qy, qz, qw):
    return np.array([
        [1-2*qy*qy-2*qz*qz, 2*qx*qy-2*qz*qw, 2*qx*qz+2*qy*qw],
        [2*qx*qy+2*qz*qw, 1-2*qx*qx-2*qz*qz, 2*qy*qz-2*qx*qw],
        [2*qx*qz-2*qy*qw, 2*qy*qz+2*qx*qw, 1-2*qx*qx-2*qy*qy]
    ])

def split_into_laps(trajectory, timestamps, dist_threshold=0.5, min_time_between_laps=5.0):
    """
    Splits trajectory into laps based on returning close to start position.
    
    trajectory: Nx2 or Nx3 array
    timestamps: list or array
    dist_threshold: distance to consider "back at start"
    min_time_between_laps: avoid false positives
    """
    laps = []
    current_lap = [0]

    start_pos = trajectory[0]
    last_lap_time = timestamps[0]

    for i in range(1, len(trajectory)):
        dist_to_start = np.linalg.norm(trajectory[i] - start_pos)

        # Detect lap completion
        if dist_to_start < dist_threshold and (timestamps[i] - last_lap_time) > min_time_between_laps:
            laps.append(current_lap)
            current_lap = []
            start_pos = trajectory[i]
            last_lap_time = timestamps[i]

        current_lap.append(i)

    if current_lap:
        laps.append(current_lap)

    return laps

#### Parameters to change ####
Plot = True
IncludeNatNav = False #Avalible for Dynamic, Spare and Ceiling
offset = np.array([0.17,0,-0.1])
test_name = "Dynamic_04_22" #LoopTest, Dynamic, Sparse, Ceiling
time_offset = 5.8 #2.78 for LoopTest, 5.0 for Dynamic, 3.0 for Sparse
startTime = 0.0


#### Main Code ####
trajectory_name = "ORB2_SLAM_dyn"
estimation = open("/home/walldenviktor/pyslam/results/"+trajectory_name+"/trajectory_online.txt", "r", encoding="utf-8")
gt = open("/home/walldenviktor/Videos/LidarData/"+test_name, "r", encoding="utf-8")
t, x, y, z, qx, qy, qz, qw, yaw = [], [], [], [], [], [], [], [], []
t_gt, x_gt, y_gt, yaw_gt = [], [], [], []
t_nn, x_nn, y_nn, yaw_nn = [], [], [], []

#Read estimate data
data = estimation.read()
lines = data.split("\n")
lines.pop(-1)

#Estimate, given in TUM format [t,x,y,z,qx,qy,qz,qw]
for line in lines:
        vals = line.split(" ")
        t.append(float(vals[0]))
        x.append(float(vals[1]))
        y.append(float(vals[2]))
        z.append(float(vals[3]))
        qx.append(float(vals[4]))
        qy.append(float(vals[5]))
        qz.append(float(vals[6]))
        qw.append(float(vals[7]))
        yaw.append(-yaw_from_cv_quaternion(float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])))
yaw = np.unwrap(yaw)

data_gt = gt.read()
lines_gt = data_gt.split("\n")
lines_gt.pop(0)

if not IncludeNatNav: #Read GT data
    for line_gt in lines_gt:
        vals = line_gt.split(" ")
        if "state" in line_gt:
            t_gt.append(float(vals[1]))
            x_gt.append(float(vals[2]))
            y_gt.append(float(vals[3]))
            yaw_gt.append(float(vals[4]))
    aligned = t_gt

else: #Read GT data and NatNav data
    natNav = open("/home/walldenviktor/Videos/"+test_name+"/natnav", "r", encoding="utf-8")
    data_nn = natNav.read()
    lines_nn = data_nn.split("\n")
    lines_nn.pop(0)

    #Reads times separately to find the aligned timestamps
    for line_gt in lines_gt:
        vals = line_gt.split(" ")
        if "state" in line_gt:
            t_gt.append(float(vals[1]))

    for line_nn in lines_nn:
        vals = line_nn.split(" ")
        if "state" in line_nn:
            t_nn.append(float(vals[1]))

    aligned = [t for t in t_gt if t in t_nn]

    for line_gt in lines_gt:
        vals = line_gt.split(" ")
        if "state" in line_gt and float(vals[1]) in aligned:
            x_gt.append(float(vals[2]))
            y_gt.append(float(vals[3]))
            yaw_gt.append(float(vals[4]))

        
    for line_nn in lines_nn:
        vals = line_nn.split(" ")
        if "state" in line_nn and float(vals[1]) in aligned:
            x_nn.append(float(vals[2]))
            y_nn.append(float(vals[3]))
            yaw_nn.append(float(vals[4]))
    yaw_nn = np.unwrap(yaw_nn)

yaw_gt = np.unwrap(yaw_gt)


matches  = associate(t, aligned, offset=time_offset, max_difference=1/41,startTime=startTime) 
est_matches = []
gt_matches = []
nn_matches = []
t_matched = []
yaw_matched = []
yaw_gt_matched = []
yaw_nn_matched =[]

#Match timestamps and do camera to body translation
for i in matches:
    j  = matches[i][1]
    yaw_matched.append(yaw[i])
    yaw_gt_matched.append(yaw_gt[j])
    gt_matches.append([x_gt[j], y_gt[j], 0.0])  # ground truth
    t_matched.append(matches[i][2])
    if IncludeNatNav:
        nn_matches.append([x_nn[j], y_nn[j], 0.0])  # NatNav
        yaw_nn_matched.append(yaw_nn[j])
    t_cam = np.array([x[i], y[i], z[i]])
    R = quat_to_R(qx[i], qy[i], qz[i], qw[i])   
    t_agv = t_cam + R @ offset
    est_matches.append([t_agv[2],-t_agv[0], 0]) 

est_arr = np.asarray(est_matches, dtype=float)
gt_arr  = np.asarray(gt_matches, dtype=float)
nn_arr  = np.asarray(nn_matches, dtype=float)


#Rotate and translate estimated trajectory to GT frame
T_gt_est, T_est_gt, is_ok = align_3d_points_with_svd(gt_arr, est_arr, find_scale=False)
est_transformed = (T_gt_est[:3, :3] @ est_arr.T).T + T_gt_est[:3, 3]

t_matched = [t - t_matched[0] for t in t_matched] #Timestamps for ploting

laps = split_into_laps(est_transformed[:, :2], t_matched,
                       dist_threshold=0.2,   # tune this!
                       min_time_between_laps=10.0)


errorX = est_transformed[:, [0]] - gt_arr[:, [0]]
errorY = est_transformed[:, [1]] - gt_arr[:, [1]]  
traj_dists = np.linalg.norm(np.column_stack((errorX, errorY)), axis=1)
rms_error = np.sqrt(np.mean(np.power(traj_dists, 2)))

#Convert yaw to be continuous and start at 0
yaw_matched_converted = []
start_angle = yaw_gt_matched[0]
yaw_matched = np.array(yaw_matched)- yaw_matched[0]
yaw_gt_matched = np.array(yaw_gt_matched)- yaw_gt_matched[0]
if IncludeNatNav:
    yaw_nn_matched = np.array(yaw_nn_matched)- yaw_nn_matched[0]

#Calculates the angle error in degrees
angle_error = []
angle_error_nn = []
for i in range(len(yaw_matched)):
    angle_error.append(math.degrees(yaw_matched[i] - yaw_gt_matched[i]))
    if IncludeNatNav:
        angle_error_nn.append(math.degrees(yaw_nn_matched[i] - yaw_gt_matched[i]))



print("Estimate errors:")
print("Max-x: %.3f Max-y: %.3f RMS: %.3f" % (max(abs(errorX))[0],
                                     max(abs(errorY))[0],
                                     rms_error))

print("Max-angle error: %.3f Average-angle error: %.3f" % (max(np.abs(angle_error)),np.mean(np.mean(angle_error))))
if IncludeNatNav:
    errorX_nn = nn_arr[:, [0]] - gt_arr[:, [0]]
    errorY_nn = nn_arr[:, [1]] - gt_arr[:, [1]]  
    traj_dists_nn = np.linalg.norm(np.column_stack((errorX_nn, errorY_nn)), axis=1)
    rms_error_nn = np.sqrt(np.mean(np.power(traj_dists_nn, 2)))
    print("NatNav errors:")
    print("Max-x: %.3f Max-y: %.3f RMS: %.3f" % (max(abs(errorX_nn))[0],
                                        max(abs(errorY_nn))[0],
                                        rms_error_nn))
    print("Max-angle error: %.3f Average-angle error: %.3f" % (max(np.abs(angle_error_nn)),np.mean(np.abs(angle_error_nn))))


#### Plotting ####
if Plot:

    # ------------------ Trajectory plot ------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.plot(est_transformed[:, 0], est_transformed[:, 1],
            label='Estimated trajectory (SLAM)', color="#34a1d4")
    ax.plot(gt_arr[:, 0], gt_arr[:, 1],
            label='Ground truth trajectory', color="#C72929DA")

    if IncludeNatNav:
        ax.plot(nn_arr[:, 0], nn_arr[:, 1],
                label='NatNav trajectory', color="#e49653", linestyle='--')

    if test_name not in ["Sparse", "Ceiling"]:
        ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('X position [m]')
    ax.set_ylabel('Y position [m]')
    ax.set_title('2D Trajectory Comparison')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best')

    fig.tight_layout()


    # ------------------ Trajectory error ------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    ax2.plot(t_matched, traj_dists,
             label='SLAM position error', color="#34a1d4")

    if IncludeNatNav:
        ax2.plot(t_matched, traj_dists_nn,
                 label='NatNav position error', color="#e49653")

    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Position error [m]')
    ax2.set_title('Position Error Over Time')
    ax2.grid(True, linestyle='--', alpha=0.4)
    ax2.legend(loc='best')

    fig2.tight_layout()


    # ------------------ Yaw plot ------------------
    fig3, ax3 = plt.subplots(figsize=(7, 4))

    ax3.plot(t_matched, yaw_matched,
             label='Estimated yaw', color="#34a1d4")
    ax3.plot(t_matched, np.unwrap(yaw_gt_matched),
             label='Ground truth yaw', color="#C72929DA")

    if IncludeNatNav:
        ax3.plot(t_matched, yaw_nn_matched,
                 label='NatNav yaw', color="#e49653")

    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Yaw angle [rad]')
    ax3.set_title('Yaw Angle Over Time')
    ax3.grid(True, linestyle='--', alpha=0.4)
    ax3.legend(loc='best')

    fig3.tight_layout()


    # ------------------ Lap-wise trajectories ------------------
    num_laps = len(laps)
    cols = 2
    rows = int(np.ceil(num_laps / cols))

    fig4, axes = plt.subplots(rows, cols, figsize=(10, 5 * rows))
    axes = axes.flatten()

    for i, lap_indices in enumerate(laps):
        ax = axes[i]

        est_lap = est_transformed[lap_indices]
        gt_lap = gt_arr[lap_indices]

        ax.plot(est_lap[:, 0], est_lap[:, 1],
                label='Estimation', color="#34a1d4")
        ax.plot(gt_lap[:, 0], gt_lap[:, 1],
                label='Ground truth', color="#C72929DA")
        
        if IncludeNatNav and len(nn_arr) > 0:
            nn_lap = nn_arr[lap_indices]
            ax.plot(nn_lap[:, 0], nn_lap[:, 1],
                    label='NatNav', color="#e49653", linestyle='--')

        ax.set_title(f'Lap {i+1}')
        ax.set_xlabel('X position [m]')
        ax.set_ylabel('Y position [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best')

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig4.suptitle('Trajectory per Lap', fontsize=14)
    fig4.tight_layout()

    plt.show()