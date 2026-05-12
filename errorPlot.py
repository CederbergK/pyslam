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

def compute_motion_metrics(traj, t):
    traj = traj[:, :2]  # use XY only
    t = np.asarray(t)

    dt = np.diff(t)
    dt[dt <= 0] = 1e-3  # safety

    x = traj[:,0]
    y = traj[:,1]

    # velocity
    vx = np.diff(x) / dt
    vy = np.diff(y) / dt
    vel = np.sqrt(vx**2 + vy**2)

    # acceleration
    ax = np.diff(vx) / dt[:-1]
    ay = np.diff(vy) / dt[:-1]
    acc = np.sqrt(ax**2 + ay**2)

    # jerk
    jx = np.diff(ax) / dt[:-2]
    jy = np.diff(ay) / dt[:-2]
    jerk = np.sqrt(jx**2 + jy**2)

    return vel, acc, jerk

def select_laps(laps, selection):
    """
    selection:
        "all"      -> all laps
        int        -> single lap (1-based index)
        (start,end)-> range inclusive, e.g. (3,6)
    """
    if selection == "all":
        return laps

    elif isinstance(selection, int):
        return [laps[selection - 1]]  # convert to 0-based index

    elif isinstance(selection, tuple) and len(selection) == 2:
        start, end = selection
        return laps[start - 1:end]  # inclusive range

    else:
        raise ValueError("Invalid lap selection")

def wrap_angle(a):
    return np.arctan2(np.sin(a), np.cos(a))

def vision_data_to_estimation(trajectory_name):
    data = open("/home/walldenviktor/pyslam/results/"+trajectory_name+"/trajectory_online.txt", "r", encoding="utf-8")
    lines = data.read().split("\n")
    lines.pop(-1)
    t, x, y, z, qx, qy, qz, qw, yaw = [], [], [], [], [], [], [], [], []
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
        yaw.append(-yaw_from_cv_quaternion(float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])))  # Why is this minus?
    yaw = np.unwrap(yaw)
    return t, x, y, z, qx, qy, qz, qw, yaw

def AGV_data_to_estimation(test_name, IncludeNatNav):
    t_gt, x_gt, y_gt, yaw_gt = [], [], [], []
    t_nn, x_nn, y_nn, yaw_nn = [], [], [], []
    encoder, gyro, gt_full = [], [], []

    gt = open("/home/walldenviktor/Videos/LidarData/"+test_name, "r", encoding="utf-8")
    data_gt = gt.read()
    lines_gt = data_gt.split("\n")
    lines_gt.pop(0)

    if not IncludeNatNav: #Read GT data, currently always extracting encoder and gyro data
        for line_gt in lines_gt:
            vals = line_gt.split(" ")
            if "state" in line_gt:
                t_gt.append(float(vals[1]))  # state, encoder and gyro have the same timestamps, so we can use the state timestamps for all
                x_gt.append(float(vals[2]))
                y_gt.append(float(vals[3]))
                yaw_gt.append(float(vals[4]))
                gt_full.append([float(vals[2]), float(vals[3]), 0.0])
            elif "enc" in line_gt and len(vals) == 6:
                encoder.append([float(vals[2]), float(vals[4])])
            elif "gyro" in line_gt:
                gyro.append(float(vals[5]))
        aligned = t_gt
        yaw_gt = np.unwrap(yaw_gt)
        return t_gt, x_gt, y_gt, yaw_gt, encoder, gyro, gt_full, aligned

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
        yaw_gt = np.unwrap(yaw_gt)    

        for line_nn in lines_nn:
            vals = line_nn.split(" ")
            if "state" in line_nn and float(vals[1]) in aligned:
                x_nn.append(float(vals[2]))
                y_nn.append(float(vals[3]))
                yaw_nn.append(float(vals[4]))
        yaw_nn = np.unwrap(yaw_nn)

        return t_gt, x_gt, y_gt, yaw_gt, x_nn, y_nn, yaw_nn, aligned
    
def EKF_fusion(IHS_matches, encoder, gyro, slam_to_encoder_idx):
    
    N = len(encoder)

    # State: [x, y, yaw]
    X = np.zeros((N, 3))
    X_pred_sampled = np.zeros((N, 3))
    P = np.eye(3) * 0.1

    # Initialize from first SLAM pose
    #X[0, 0:2] = est_arr[0, 0:2]
    #X[0, 2] = yaw_matched[0]
    X[0, 0:2] = [0,0]
    X[0, 2] = 0

    # Noise matrices (TUNE THESE!)
    Q = np.diag([0.1, 0.1, 0.1])   # process noise
    R = np.diag([1, 1, 1])   # measurement noise

    slam_idx = 0

    for k in range(1, N):

        dt = t_gt[k] - t_gt[k-1]
        if dt <= 0:
            dt = 1e-3

        # -------- PREDICTION --------
        yaw = X[k-1, 2]

        # Get encoder velocities
        v_l = encoder[k][0]
        v_r = encoder[k][1]

        v = (v_r + v_l) / 2.0

        # Angular velocity fusion
        wheel_base = 0.40
        omega_enc = (v_r - v_l) / wheel_base
        omega_gyro = gyro[k]

        alpha = 0.29
        omega = alpha * omega_gyro + (1 - alpha) * omega_enc

        theta_new = yaw + omega * dt

        # Motion model
        if abs(omega) > 1e-6:
            x_pred = X[k-1, 0] + v/omega * (np.sin(theta_new) - np.sin(yaw))
            y_pred = X[k-1, 1] + v/omega * (-np.cos(theta_new) + np.cos(yaw))
        else:
            x_pred = X[k-1, 0] + v * np.cos(yaw) * dt
            y_pred = X[k-1, 1] + v * np.sin(yaw) * dt

        yaw_pred = theta_new

        X_pred = np.array([x_pred, y_pred, yaw_pred])
        X_pred_sampled[k] = X_pred

        # Jacobian
        F = np.array([
            [1, 0, -v * dt * np.sin(yaw)],
            [0, 1,  v * dt * np.cos(yaw)],
            [0, 0, 1]
        ])

        P = F @ P @ F.T + Q
        X[k] = X_pred

        # -------- UPDATE (SLAM measurement) --------
        if slam_idx < len(slam_to_encoder_idx):
            slam_i, enc_j = slam_to_encoder_idx[slam_idx]
            if i == enc_j:
                z = np.array([IHS_matches[slam_idx, 0], IHS_matches[slam_idx, 1], yaw_IHS[slam_idx]])

                H = np.eye(3)

                y_residual = z - X_pred

                # normalize angle
                y_residual[2] = np.arctan2(np.sin(y_residual[2]), np.cos(y_residual[2]))

                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)

                X[k] = X_pred + K @ y_residual
                P = (np.eye(3) - K @ H) @ P

                slam_idx += 1

    # Replace SLAM trajectory with fused one
    IHS_fused = np.zeros((N, 3))
    IHS_fused[:, 0:2] = X[:, 0:2]
    yaw_fused = X[:, 2]
    return IHS_fused, yaw_fused

# ============================================= PARAMETERS TO CHANGE ====================================================
Plot = True
IncludeNatNav = True #Avalible for Dynamic, Spare and Ceiling
add_EKF = False
debug_plot = False

lap_selection = (3,6)  # <-- int, (start,end) or "all"
offset = np.array([0.17,0,-0.1])
time_offset = 5.79 #2.78 for LoopTest, 5.0 for Dynamic, 3.0 for Sparse, 5.79 for New Dynamic
startTime = 0.0

# ============================================= Extracting estimations ====================================================
test_name = "Dynamic_04_22" #LoopTest, Dynamic, Sparse, Ceiling, Dynamic_04_22

#### Read and extract estimate data, given in TUM format [t,x,y,z,qx,qy,qz,qw] ####
t_IHS, x_IHS, y_IHS, z_IHS, qx_IHS, qy_IHS, qz_IHS, qw_IHS, yaw_IHS = vision_data_to_estimation("IHS_test")
t_OSS, x_OSS, y_OSS, z_OSS, qx_OSS, qy_OSS, qz_OSS, qw_OSS, yaw_OSS = vision_data_to_estimation("Loc-Map_test")
t_SLAM, x_SLAM, y_SLAM, z_SLAM, qx_SLAM, qy_SLAM, qz_SLAM, qw_SLAM, yaw_SLAM = vision_data_to_estimation("SLAM-map_test")


if IncludeNatNav:
    t_gt, x_gt, y_gt, yaw_gt, x_nn, y_nn, yaw_nn, aligned = AGV_data_to_estimation(test_name, IncludeNatNav)
else:
    t_gt, x_gt, y_gt, yaw_gt, encoder, gyro, gt_full, aligned = AGV_data_to_estimation(test_name, IncludeNatNav)
    gt_full  = np.asarray(gt_full, dtype=float)


# ============================================= Associate timestamps and transform camera to robot frame ====================================================
IHS_matches, OSS_matches, SLAM_matches, gt_matches, nn_matches = [], [], [], [], []
yaw_IHS_matches, yaw_OSS_matches, yaw_SLAM_matches, yaw_gt_matched, yaw_nn_matched = [], [], [], [], []

t_matched = []

# EKF stuff
encoder_matched = []
gyro_matched = []
slam_to_encoder_idx = []

#Match timestamps and do camera to body translation
matches  = associate(t_IHS, aligned, offset=time_offset, max_difference=1/41,startTime=startTime) 
for i in matches:
    j  = matches[i][1]
    t_matched.append(matches[i][2])

    gt_matches.append([x_gt[j], y_gt[j], 0.0])  # ground truth, add gt_full!

    t_cam_IHS = np.array([x_IHS[i], y_IHS[i], z_IHS[i]])
    t_cam_OSS = np.array([x_OSS[i], y_OSS[i], z_OSS[i]])
    t_cam_SLAM = np.array([x_SLAM[i], y_SLAM[i], z_SLAM[i]])

    # Apply camera to body transformation
    R_IHS = quat_to_R(qx_IHS[i], qy_IHS[i], qz_IHS[i], qw_IHS[i])
    R_OSS = quat_to_R(qx_OSS[i], qy_OSS[i], qz_OSS[i], qw_OSS[i])
    R_SLAM = quat_to_R(qx_SLAM[i], qy_SLAM[i], qz_SLAM[i], qw_SLAM[i])
    t_agv_IHS = t_cam_IHS + R_IHS @ offset
    t_agv_OSS = t_cam_OSS + R_OSS @ offset
    t_agv_SLAM = t_cam_SLAM + R_SLAM @ offset

    # Apply rotation of camera frame into robot frame manually
    IHS_matches.append([t_agv_IHS[2],-t_agv_IHS[0], 0]) 
    OSS_matches.append([t_agv_OSS[2], -t_agv_OSS[0], 0])
    SLAM_matches.append([t_agv_SLAM[2], -t_agv_SLAM[0], 0])

    # Yaw matches
    yaw_IHS_matches.append(yaw_IHS[i])
    yaw_OSS_matches.append(yaw_OSS[i])
    yaw_SLAM_matches.append(yaw_SLAM[i])
    yaw_gt_matched.append(yaw_gt[j])

    # Conditions to include NatNav and EKF
    if IncludeNatNav:
        nn_matches.append([x_nn[j], y_nn[j], 0.0])  # NatNav
        yaw_nn_matched.append(yaw_nn[j])
    elif add_EKF:
        slam_to_encoder_idx.append((i, j))
        encoder_matched.append(encoder[j])
        gyro_matched.append(gyro[j])


IHS_matches = np.asarray(IHS_matches, dtype=float)
OSS_matches = np.asarray(OSS_matches, dtype=float)
SLAM_matches = np.asarray(SLAM_matches, dtype=float)

gt_matches  = np.asarray(gt_matches, dtype=float)
nn_matches  = np.asarray(nn_matches, dtype=float)

# ============================================= EKF Fusion but yaw_fused needs fixing ====================================================
if add_EKF:
    IHS_fused, yaw_fused = EKF_fusion(IHS_matches, encoder, gyro, slam_to_encoder_idx) 
    IHS_for_estimation = IHS_fused
    gt_for_estimation = gt_full
    t_matched = [t - t_matched[0] for t in t_gt] #Timestamps for ploting
    t_matched = np.array(t_matched)
    yaw_gt_matched = yaw_gt
else:
    IHS_for_estimation = IHS_matches
    gt_for_estimation = gt_matches
    t_matched = [t - t_matched[0] for t in t_matched] #Timestamps for ploting
    t_matched = np.array(t_matched)

# ============================================= Rotate and translate estimations to robot world ====================================================
T_gt_IHS, T_IHS_gt, _ = align_3d_points_with_svd(gt_for_estimation, IHS_for_estimation, find_scale=False)
T_gt_OSS, T_OSS_gt, _ = align_3d_points_with_svd(gt_for_estimation, OSS_matches, find_scale=False)
T_gt_SLAM, T_SLAM_gt, _ = align_3d_points_with_svd(gt_for_estimation, SLAM_matches, find_scale=False)

IHS = (T_gt_IHS[:3, :3] @ IHS_for_estimation.T).T + T_gt_IHS[:3, 3]
OSS = (T_gt_OSS[:3, :3] @ OSS_matches.T).T + T_gt_OSS[:3, 3]
SLAM = (T_gt_SLAM[:3, :3] @ SLAM_matches.T).T + T_gt_SLAM[:3, 3]

# ============================================= Lap splitting and selection ====================================================
laps = split_into_laps(IHS[:, :2], t_matched, dist_threshold=0.2, min_time_between_laps=10.0)  # Only splitting with IHS, as the others are very similar.
laps_to_plot = select_laps(laps, lap_selection)
laps_for_eval = np.concatenate([np.array(lap) for lap in laps_to_plot])

IHS = IHS[laps_for_eval]
OSS = OSS[laps_for_eval]
SLAM = SLAM[laps_for_eval]
gt_for_eval = gt_for_estimation[laps_for_eval]
t_matched = t_matched[laps_for_eval]

yaw_IHS = np.array(yaw_IHS_matches)
yaw_OSS = np.array(yaw_OSS_matches)
yaw_SLAM = np.array(yaw_SLAM_matches)
yaw_gt_matched = np.array(yaw_gt_matched)

yaw_IHS = yaw_IHS[laps_for_eval]
yaw_OSS = yaw_OSS[laps_for_eval]
yaw_SLAM = yaw_SLAM[laps_for_eval]
yaw_gt_matched = yaw_gt_matched[laps_for_eval]

# ============================================= Global translation errors ====================================================
errorX_IHS = IHS[:, [0]] - gt_for_eval[:, [0]]
errorY_IHS = IHS[:, [1]] - gt_for_eval[:, [1]]
errorX_OSS = OSS[:, [0]] - gt_for_eval[:, [0]]
errorY_OSS = OSS[:, [1]] - gt_for_eval[:, [1]]
errorX_SLAM = SLAM[:, [0]] - gt_for_eval[:, [0]]
errorY_SLAM = SLAM[:, [1]] - gt_for_eval[:, [1]]

IHS_ATE = np.linalg.norm(np.column_stack((errorX_IHS, errorY_IHS)), axis=1)
OSS_ATE = np.linalg.norm(np.column_stack((errorX_OSS, errorY_OSS)), axis=1)
SLAM_ATE = np.linalg.norm(np.column_stack((errorX_SLAM, errorY_SLAM)), axis=1)

IHS_rms_error = np.sqrt(np.mean(np.power(IHS_ATE, 2)))
OSS_rms_error = np.sqrt(np.mean(np.power(OSS_ATE, 2)))
SLAM_rms_error = np.sqrt(np.mean(np.power(SLAM_ATE, 2)))

IHS_max_error = np.max(IHS_ATE)
OSS_max_error = np.max(OSS_ATE)
SLAM_max_error = np.max(SLAM_ATE)

print("\n --- Translation and Rotation Metrics ---")
#print("Max-x: %.3f Max-y: %.3f RMS: %.3f" % (max(abs(errorX))[0],max(abs(errorY))[0],rms_error))
print("IHS Max Error: %.3f IHS RMSE: %.3f" % (IHS_max_error,IHS_rms_error))
print("OSS Max Error: %.3f OSS RMSE: %.3f" % (OSS_max_error,OSS_rms_error))
print("SLAM Max Error: %.3f SLAM RMSE: %.3f" % (SLAM_max_error,SLAM_rms_error))


# ============================================= Global angle errors ====================================================
#Convert yaw to be continuous and start at 0
yaw_matched_converted = []
start_angle = yaw_gt_matched[0]
yaw_IHS = np.array(yaw_IHS)- yaw_IHS[0]
yaw_OSS = np.array(yaw_OSS)- yaw_OSS[0]
yaw_SLAM = np.array(yaw_SLAM)- yaw_SLAM[0]
yaw_gt_matched = np.array(yaw_gt_matched)- yaw_gt_matched[0]
if IncludeNatNav:
    yaw_nn_matched = np.array(yaw_nn_matched)
    yaw_nn_matched = yaw_nn_matched[laps_for_eval]
    yaw_nn_matched = np.array(yaw_nn_matched)- yaw_nn_matched[0]

#Calculates the angle error in degrees
angle_error_IHS, angle_error_OSS, angle_error_SLAM = [], [], []
angle_error_nn = []
for i in range(len(yaw_IHS)):
    angle_error_IHS.append(math.degrees(yaw_IHS[i] - yaw_gt_matched[i]))
    angle_error_OSS.append(math.degrees(yaw_OSS[i] - yaw_gt_matched[i]))
    angle_error_SLAM.append(math.degrees(yaw_SLAM[i] - yaw_gt_matched[i]))
    if IncludeNatNav:
        angle_error_nn.append(math.degrees(yaw_nn_matched[i] - yaw_gt_matched[i]))

print("IHS Max-angle error: %.3f IHS Average-angle error: %.3f" % (max(np.abs(angle_error_IHS)),np.mean(np.mean(angle_error_IHS))))
print("OSS Max-angle error: %.3f OSS Average-angle error: %.3f" % (max(np.abs(angle_error_OSS)),np.mean(np.mean(angle_error_OSS))))
print("SLAM Max-angle error: %.3f SLAM Average-angle error: %.3f" % (max(np.abs(angle_error_SLAM)),np.mean(np.mean(angle_error_SLAM))))
if IncludeNatNav:
    nn_arr = nn_matches[laps_for_eval]
    errorX_nn = nn_arr[:, [0]] - gt_for_eval[:, [0]]
    errorY_nn = nn_arr[:, [1]] - gt_for_eval[:, [1]]  
    traj_dists_nn = np.sqrt(errorX_nn**2 + errorY_nn**2)
    rms_error_nn = np.sqrt(np.mean(np.power(traj_dists_nn, 2)))
    max_pos_error_nn = np.max(traj_dists_nn)
    print("\n --- NatNav Translation and Rotation Metrics ---")
    print("Max Error: %.3f RMS: %.3f" % (max_pos_error_nn, rms_error_nn))
    print("Max-angle error: %.3f Average-angle error: %.3f" % (max(np.abs(angle_error_nn)),np.mean(np.abs(angle_error_nn))))


# ============================================= Longitudinal/Lateral translation errors ====================================================
IHS_error_robot = []
OSS_error_robot = []
SLAM_error_robot = []
NatNav_error_robot = []

for i in range(len(IHS)):

    # global error vector
    IHS_dx = IHS[i,0] - gt_for_eval[i,0]
    IHS_dy = IHS[i,1] - gt_for_eval[i,1]
    OSS_dx = OSS[i,0] - gt_for_eval[i,0]
    OSS_dy = OSS[i,1] - gt_for_eval[i,1]
    SLAM_dx = SLAM[i,0] - gt_for_eval[i,0]
    SLAM_dy = SLAM[i,1] - gt_for_eval[i,1]
    if IncludeNatNav:
        nn_dx = nn_arr[i,0] - gt_for_eval[i,0]
        nn_dy = nn_arr[i,1] - gt_for_eval[i,1]

    theta = yaw_gt_matched[i]

    # world -> robot rotation
    c = np.cos(theta)
    s = np.sin(theta)

    R_world_robot = np.array([
        [ c, s],
        [-s, c]
    ])

    IHS_err_robot = R_world_robot @ np.array([IHS_dx, IHS_dy])
    OSS_err_robot = R_world_robot @ np.array([OSS_dx, OSS_dy])
    SLAM_err_robot = R_world_robot @ np.array([SLAM_dx, SLAM_dy])
        
    IHS_error_robot.append(IHS_err_robot)
    OSS_error_robot.append(OSS_err_robot)
    SLAM_error_robot.append(SLAM_err_robot)

    if IncludeNatNav:
        nn_err_robot = R_world_robot @ np.array([nn_dx, nn_dy])
        NatNav_error_robot.append(nn_err_robot)

IHS_error_robot = np.array(IHS_error_robot)
OSS_error_robot = np.array(OSS_error_robot)
SLAM_error_robot = np.array(SLAM_error_robot)

IHS_longitudinal_error = IHS_error_robot[:,0]
IHS_lateral_error = IHS_error_robot[:,1]
OSS_longitudinal_error = OSS_error_robot[:,0]
OSS_lateral_error = OSS_error_robot[:,1]
SLAM_longitudinal_error = SLAM_error_robot[:,0]
SLAM_lateral_error = SLAM_error_robot[:,1]

IHS_long_rmse = np.sqrt(np.mean(IHS_longitudinal_error**2))
IHS_lat_rmse  = np.sqrt(np.mean(IHS_lateral_error**2))
OSS_long_rmse = np.sqrt(np.mean(OSS_longitudinal_error**2))
OSS_lat_rmse  = np.sqrt(np.mean(OSS_lateral_error**2))
SLAM_long_rmse = np.sqrt(np.mean(SLAM_longitudinal_error**2))
SLAM_lat_rmse  = np.sqrt(np.mean(SLAM_lateral_error**2))

print("\n--- Long/Lat Metrics ---")
print("IHS Max longitudinal error: %.3f m" %
      np.max(np.abs(IHS_longitudinal_error)))
print("IHS Max lateral error: %.3f m" %
      np.max(np.abs(IHS_lateral_error)))
print("OSS Max longitudinal error: %.3f m" %
      np.max(np.abs(OSS_longitudinal_error)))
print("OSS Max lateral error: %.3f m" %
      np.max(np.abs(OSS_lateral_error)))
print("SLAM Max longitudinal error: %.3f m" %
      np.max(np.abs(SLAM_longitudinal_error)))
print("SLAM Max lateral error: %.3f m" %
      np.max(np.abs(SLAM_lateral_error)))

if IncludeNatNav:
    NatNav_error_robot = np.array(NatNav_error_robot)
    nn_longitudinal_error = NatNav_error_robot[:,0]
    nn_lateral_error = NatNav_error_robot[:,1]
    nn_long_rmse = np.sqrt(np.mean(nn_longitudinal_error**2))
    nn_lat_rmse  = np.sqrt(np.mean(nn_lateral_error**2))
    print("\n--- NatNav Long/Lat Metrics ---")
    print("NatNav Max longitudinal error: %.3f m" % np.max(np.abs(nn_longitudinal_error)))
    print("NatNav Max lateral error: %.3f m" % np.max(np.abs(nn_lateral_error)))


# ============================================= Velocity and acceleration errors ====================================================
IHS_vel_est, IHS_acc_est, _ = compute_motion_metrics(IHS, t_matched)
OSS_vel_est, OSS_acc_est, _ = compute_motion_metrics(OSS, t_matched)
SLAM_vel_est, SLAM_acc_est, _ = compute_motion_metrics(SLAM, t_matched)
vel_gt,  acc_gt,  _  = compute_motion_metrics(gt_for_eval, t_matched)

# Velocity RMSE
min_len = min(len(IHS_vel_est), len(vel_gt))
IHS_vel_rmse = np.sqrt(np.mean((IHS_vel_est[:min_len] - vel_gt[:min_len])**2))
OSS_vel_rmse = np.sqrt(np.mean((OSS_vel_est[:min_len] - vel_gt[:min_len])**2))
SLAM_vel_rmse = np.sqrt(np.mean((SLAM_vel_est[:min_len] - vel_gt[:min_len])**2))

# Acceleration std
IHS_acc_rmse = np.sqrt(np.mean((IHS_acc_est[:min_len] - acc_gt[:min_len])**2))
OSS_acc_rmse = np.sqrt(np.mean((OSS_acc_est[:min_len] - acc_gt[:min_len])**2))
SLAM_acc_rmse = np.sqrt(np.mean((SLAM_acc_est[:min_len] - acc_gt[:min_len])**2))

print("\n--- SLAM Motion Metrics ---")
print("IHS Velocity RMSE: %.3f m/s" % IHS_vel_rmse)
print("IHS Acceleration RMSE: %.3f m/s²" % IHS_acc_rmse)
print("OSS Velocity RMSE: %.3f m/s" % OSS_vel_rmse)
print("OSS Acceleration RMSE: %.3f m/s²" % OSS_acc_rmse)
print("SLAM Velocity RMSE: %.3f m/s" % SLAM_vel_rmse)
print("SLAM Acceleration RMSE: %.3f m/s²" % SLAM_acc_rmse)

if IncludeNatNav and len(nn_arr) > 0:
    vel_nn, acc_nn, _ = compute_motion_metrics(nn_arr, t_matched)

    min_len_nn = min(len(vel_nn), len(vel_gt))
    vel_rmse_nn = np.sqrt(np.mean((vel_nn[:min_len_nn] - vel_gt[:min_len_nn])**2))

    acc_rmse_nn = np.sqrt(np.mean((acc_nn[:min_len_nn] - acc_gt[:min_len_nn])**2))

    print("\n--- NatNav Motion Metrics ---")
    print("Velocity RMSE: %.3f m/s" % vel_rmse_nn)
    print("Acceleration RMSE: %.3f m/s²" % acc_rmse_nn)

# ============================================= Plotting ====================================================
if Plot:

    # ------------------ Trajectory plot ------------------
    num_laps = len(laps_to_plot)
    if num_laps == 1:
        cols, rows = 1, 1
    elif num_laps <= 3:
        cols, rows = num_laps, 1
    else:
        cols = 2
        rows = int(np.ceil(num_laps / cols))

    fig1, axes = plt.subplots(rows, cols, figsize=(5*cols, 5 * rows))
    if num_laps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    selected_indices = [laps.index(l) for l in laps_to_plot]

    # Create remapped local indices
    offset = 0

    for plot_idx, lap_indices in enumerate(laps_to_plot):

        lap_number = selected_indices[plot_idx] + 1
        ax = axes[plot_idx]

        lap_len = len(lap_indices)

        # local indices into already-filtered arrays
        local_idx = np.arange(offset, offset + lap_len)

        IHS_plot = IHS[local_idx]
        gt_t = gt_for_eval[local_idx]

        offset += lap_len
        ax.plot(gt_t[:, 0], gt_t[:, 1],
                label='Ground truth', color="#C72929DA")
        ax.plot(IHS_plot[:, 0], IHS_plot[:, 1],
                label='IHS', color="#34a1d4")
        OSS_plot = OSS[local_idx]
        ax.plot(OSS_plot[:, 0], OSS_plot[:, 1],
                label='OSS', color="#22d148")
        SLAM_plot = SLAM[local_idx]
        ax.plot(SLAM_plot[:, 0], SLAM_plot[:, 1],
                label='SLAM', color="#7708D1DA", linestyle='--')

        if IncludeNatNav and len(nn_arr) > 0:
            nn_lap = nn_arr[local_idx]
            ax.plot(nn_lap[:, 0], nn_lap[:, 1],
                    label='NatNav', color="#e49653", linestyle='--')

        ax.set_title(f'Lap {lap_number}')
        ax.set_xlabel('X position [m]')
        ax.set_ylabel('Y position [m]')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='best')

    # Hide unused subplots
    for j in range(num_laps, len(axes)):
        axes[j].axis('off')

    fig1.suptitle('Trajectory per Lap', fontsize=14)
    fig1.tight_layout()


    # ------------------ Trajectory error ------------------
    fig2, ax2 = plt.subplots(figsize=(7, 4))

    ax2.plot(t_matched, IHS_ATE,
             label='IHS position error', color="#34a1d4")
    
    ax2.plot(t_matched, OSS_ATE,
            label='OSS position error', color="#22d148")
    
    ax2.plot(t_matched, SLAM_ATE,
            label='SLAM position error', color="#7708D1DA")

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

    ax3.plot(t_matched, yaw_IHS,
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

    # ------------------ Angle error plot ------------------
    fig4, ax4 = plt.subplots(figsize=(7, 4))    
    ax4.plot(t_matched, angle_error_IHS,
             label='IHS angle error', color="#34a1d4")
    ax4.plot(t_matched, angle_error_OSS,
             label='OSS angle error', color="#22d148")
    ax4.plot(t_matched, angle_error_SLAM,
             label='SLAM angle error', color="#7708D1DA")   
    if IncludeNatNav:
        ax4.plot(t_matched, angle_error_nn,
                 label='NatNav angle error', color="#e49653")

    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Angle error [rad]')
    ax4.set_title('Angle Error Over Time')
    ax4.grid(True, linestyle='--', alpha=0.4)
    ax4.legend(loc='best')

    fig4.tight_layout()

    plt.show()

if debug_plot:
    plt.figure()
    #plt.plot(gt_matches[:, 0], gt_matches[:, 1], label='GT')
    plt.plot([p[0] for p in gt_matches],
            [p[1] for p in gt_matches], label='AGV Sampled', color="r")
    plt.plot([p[2] for p in t_agv_sampled],
            [p[0] for p in t_agv_sampled], color="g")
    plt.plot([p[2] for p in t_cam_sampled],
            [p[0] for p in t_cam_sampled], color="k")
    plt.plot([p[0] for p in est_matches],
            [p[1] for p in est_matches], color="y")
    plt.plot(est_transformed[:, 0], est_transformed[:, 1], color="b")
    plt.show()

    plt.figure()
    plt.plot(t[1:], vel_est, label="Estimated speed")
    plt.plot(t[1:], vel_gt, label="GT speed")
    plt.legend()
    plt.title("Speed comparison")
    plt.grid()
    plt.show()