import matplotlib.pyplot as plt
import numpy as np
import os
import math

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

test_name = "Dynamic_04_22" #LoopTest, Dynamic, Sparse, Ceiling
gt = open("/home/walldenviktor/Videos/LidarData/"+test_name, "r", encoding="utf-8")

encoder = []
gyro = []
aligned = []
gt_full = []
time_offset = 5.79
startTime = 0.0

# Ground truth data format:

data_gt = gt.read()
lines_gt = data_gt.split("\n")
lines_gt.pop(0)

for line_gt in lines_gt:
    vals = line_gt.split(" ")
    if "state" in line_gt:
        gt_full.append([float(vals[2]), float(vals[3]), 0.0])
    if "enc" in line_gt and len(vals) == 6:
        t = float(vals[1])
        v_l = float(vals[2])
        v_r = float(vals[4])
        encoder.append([t, v_l, v_r])
        aligned.append(t)
    elif "gyro" in line_gt:
        gyro.append(float(vals[5]))

# Estimation data format:
trajectory_name = "IHS_test"
estimation = open("/home/walldenviktor/pyslam/results/"+trajectory_name+"/trajectory_online.txt", "r", encoding="utf-8")
data = estimation.read()
lines = data.split("\n")
lines.pop(-1)

#Estimate, given in TUM format [t,x,y,z,qx,qy,qz,qw]
t_est = []
for line in lines:
        vals = line.split(" ")
        t_est.append(float(vals[0]))


matches  = associate(t_est, aligned, offset=time_offset, max_difference=1/41,startTime=startTime) 
t_matched = []
encoder_matched = []
gyro_matched = []


#Match timestamps and do camera to body translation
for i in matches:
    j  = matches[i][1]
    t_matched.append(matches[i][2])
    encoder_matched.append(encoder[j])
    gyro_matched.append(gyro[j])

x, y, theta = 0.0, 0.0, 0.0
traj = [[0,0,0]]
omega_encoders = []

wheelbase = 0.40  # <-- YOU MUST SET THIS
alpha = 0.29

for i in range(1, len(encoder)):
    t_prev, vl_prev, vr_prev = encoder[i-1]
    t, vl, vr = encoder[i]

    dt = t - t_prev

    v = (vr + vl) / 2.0
    omega_encoders.append((vr - vl) / wheelbase)
    omega_gyro = gyro[i]
    omega = alpha * omega_gyro + (1 - alpha) * omega_encoders[-1]
    theta_new = theta + omega * dt

    if abs(omega) > 1e-6:
        x += v/omega * (math.sin(theta_new) - math.sin(theta))
        y += v/omega * (-math.cos(theta_new) + math.cos(theta))
    else:
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt

    theta = theta_new
    traj.append([x, y,0])

traj = np.array(traj)
plt.figure(figsize=(12, 6))
plt.plot(traj[:,0], traj[:,1])
plt.axis('equal')
plt.figure(figsize=(12, 6))
plt.plot([e[1] for e in encoder])
plt.xlabel("Left Wheel Velocity")
plt.figure(figsize=(12, 6))
plt.plot([e[2] for e in encoder])
plt.xlabel("Right Wheel Velocity")
plt.figure(figsize=(12, 6))
plt.plot([g for g in gyro])
plt.plot([g for g in omega_encoders])
plt.show()

gt_full  = np.asarray(gt_full, dtype=float)
T_gt_est, T_est_gt, is_ok = align_3d_points_with_svd(gt_full, traj, find_scale=False)
est_transformed = (T_gt_est[:3, :3] @ traj.T).T + T_gt_est[:3, 3]

errorX = est_transformed[:, [0]] - gt_full[:, [0]]
errorY = est_transformed[:, [1]] - gt_full[:, [1]]  
traj_dists = np.linalg.norm(np.column_stack((errorX, errorY)), axis=1)
rms_error = np.sqrt(np.mean(np.power(traj_dists, 2)))

print("Estimate errors:")
print("Max-x: %.3f Max-y: %.3f RMS: %.3f" % (max(abs(errorX))[0],
                                     max(abs(errorY))[0],
                                     rms_error))

fig, ax = plt.subplots(figsize=(7, 6))

ax.plot(est_transformed[:, 0], est_transformed[:, 1],
        label='Estimated trajectory (SLAM)', color="#34a1d4")
ax.plot(gt_full[:, 0], gt_full[:, 1],
        label='Ground truth trajectory', color="#C72929DA")
plt.show()