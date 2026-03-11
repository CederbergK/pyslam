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

#### Main code ####
Plot = False
IncludeNatNav = True
offset = np.array([0.04,0.0,-0.06]) 
test_name = "Dynamic"

estimation = open("/home/albincederberg/pyslam/results/"+test_name+"/trajectory_online.txt", "r", encoding="utf-8")
gt = open("/home/albincederberg/Videos/"+test_name+"/groundtruth.txt", "r", encoding="utf-8")
natNav = open("/home/albincederberg/Videos/"+test_name+"/natnav", "r", encoding="utf-8")

t, x, y, z = [], [], [], []
t_gt, x_gt, y_gt, z_gt ,q_w= [], [], [], [], []
t_nn, x_nn, y_nn = [], [], []
#Read estimate data
data = estimation.read()
lines = data.split("\n")
lines.pop(-1)

for line in lines:
        vals = line.split(" ")
        t.append(float(vals[0]))
        x.append(float(vals[1]))
        z.append(float(vals[3]))

data_gt = gt.read()
lines_gt = data_gt.split("\n")
lines_gt.pop(-1)

if IncludeNatNav:
    data_nn = natNav.read()
    lines_nn = data_nn.split("\n")
    lines_nn.pop(0)

    for line_gt in lines_gt:
        vals = line_gt.split(" ")
        t_gt.append(float(vals[0]))
    for line_nn in lines_nn:
        vals = line_nn.split(" ")
        if "state" in line_nn:
            t_nn.append(float(vals[1]))
    aligned = [t for t in t_gt if t in t_nn]

    for line_gt in lines_gt:
        vals = line_gt.split(" ")
        if float(vals[0]) in aligned:
            x_gt.append(float(vals[1]))
            y_gt.append(float(vals[2]))
            q_w.append(float(vals[7]))
    for line_nn in lines_nn:
        vals = line_nn.split(" ")
        if "state" in line_nn and float(vals[1]) in aligned:
            x_nn.append(float(vals[2]))
            y_nn.append(float(vals[3]))
else: #Read GT data
     for line in lines:
        vals = line.split(" ")
        t_gt.append(float(vals[0]))
        x_gt.append(float(vals[1]))
        y_gt.append(float(vals[2]))
        q_w.append(float(vals[7]))


matches  = associate(t, aligned, offset=4.6, max_difference=1/41,startTime=150) #2.78 for LoopTest, 4.5 for Dynamic
est_matches = []
gt_matches = []
nn_matches = []
t_matched = []


#Match timestamps and do camera to body translation
for i in matches:
    j  = matches[i][1]
    yaw = -2 * math.acos(q_w[j])
    x_corr = offset[0] * math.sin(yaw) + offset[2] * math.cos(yaw) #Corrections
    z_corr = offset[0] * math.cos(yaw) - offset[2] * math.sin(yaw)
    est_matches.append([x[i]+x_corr, 0.0, z[i]+z_corr])    # estimated
    gt_matches.append([x_gt[j], y_gt[j], 0.0])  # ground truth
    nn_matches.append([x_nn[j], y_nn[j], 0.0])  # NatNav
    t_matched.append(matches[i][2])

est_arr = np.asarray(est_matches, dtype=float)
gt_arr  = np.asarray(gt_matches, dtype=float)
nn_arr  = np.asarray(nn_matches, dtype=float)


#Rotate and translate estimated trajectory to GT frame
T_gt_est, T_est_gt, is_ok = align_3d_points_with_svd(gt_arr, est_arr, find_scale=False)
est_transformed = (T_gt_est[:3, :3] @ est_arr.T).T + T_gt_est[:3, 3]

t_matched = [t - t_matched[0] for t in t_matched] #Timestamps for ploting


errorX = est_transformed[:, [0]] - gt_arr[:, [0]]
errorY = est_transformed[:, [1]] - gt_arr[:, [1]]  
traj_dists = np.linalg.norm(np.column_stack((errorX, errorY)), axis=1)
rms_error = np.sqrt(np.mean(np.power(traj_dists, 2)))
print("Estimate errors:")
print("Max-x: %.3f Max-y: %.3f RMS: %.3f" % (max(abs(errorX))[0],
                                     max(abs(errorY))[0],
                                     rms_error))


errorX_nn = nn_arr[:, [0]] - gt_arr[:, [0]]
errorY_nn = nn_arr[:, [1]] - gt_arr[:, [1]]  
traj_dists_nn = np.linalg.norm(np.column_stack((errorX_nn, errorY_nn)), axis=1)
rms_error_nn = np.sqrt(np.mean(np.power(traj_dists_nn, 2)))
print("NatNav errors:")
print("Max-x: %.3f Max-y: %.3f RMS: %.3f" % (max(abs(errorX_nn))[0],
                                     max(abs(errorY_nn))[0],
                                     rms_error_nn))

####Plotting####
if Plot:

    fig, ax = plt.subplots(figsize=(7, 6))  # 2D axes

    ax.plot(est_transformed[:, 0], est_transformed[:, 1], label='Estimated trajectory', color='#1f77b4')
    ax.plot(gt_arr[:, 0],  gt_arr[:, 1],  label='Ground truth trajectory', color='#2ca02c')
    ax.plot(nn_arr[:, 0],  nn_arr[:, 1],  label='NatNav trajectory', color='#ff7f0e')

    ax.set_aspect('equal', adjustable='box')  # keep metric aspect ratio
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_title('Trajectories (2D)')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    fig.tight_layout()



    ax2 = plt.figure().add_subplot()
    ax2.plot(t_matched, traj_dists, label='Trajectory error')
    ax2.plot(t_matched, traj_dists_nn, label='NatNav error')

    ax2.set_xlabel('Time')
    ax2.set_ylabel('Error')
    ax2.set_title('Trajectory error over time')
    plt.ylim(0, 0.7)
    plt.show()