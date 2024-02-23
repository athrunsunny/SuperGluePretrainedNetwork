import copy
import os

import cv2
import numpy as np


def find_feature_matches(img_1, img_2):
    # 特征点检测与匹配
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(img_1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img_2, None)

    # 创建FLANN匹配器
    flann = cv2.FlannBasedMatcher()

    # 使用FLANN匹配器进行特征匹配
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # 选择良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    # good_matches = sorted(good_matches, key=lambda x: x.distance)
    # matcher = cv2.BFMatcher()
    # matches = matcher.match(descriptors1, descriptors2)
    # matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点对的关键点坐标
    # points1 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # points2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    points1 = keypoints1
    points2 = keypoints2
    return points1, points2, good_matches


def pose_estimation_2d2d(keypoints_1, keypoints_2, matches, camera_matrix):
    # points1 = np.array([keypoints_1[match.queryIdx].pt for match in matches])
    # points2 = np.array([keypoints_2[match.trainIdx].pt for match in matches])
    points1 = keypoints_1
    points2 = keypoints_2
    fundamental_matrix, mask = cv2.findFundamentalMat(points1, points2, cv2.FM_RANSAC)

    points1 = points1[mask.ravel() == 1]
    points2 = points2[mask.ravel() == 1]

    essential_matrix, _ = cv2.findEssentialMat(points1, points2, camera_matrix)

    _, R, t, _ = cv2.recoverPose(essential_matrix, points1, points2, camera_matrix)

    return R, t


def pixel2cam(p, K):
    return np.array([(p[0] - K[0, 2]) / K[0, 0], (p[1] - K[1, 2]) / K[1, 1]], dtype=np.float32)


def triangulation(keypoints_1, keypoints_2, matches, R, t, K):
    T1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0]], dtype=np.float32)

    T2 = np.array([[R[0, 0], R[0, 1], R[0, 2], t[0, 0]],
                   [R[1, 0], R[1, 1], R[1, 2], t[1, 0]],
                   [R[2, 0], R[2, 1], R[2, 2], t[2, 0]]], dtype=np.float32)

    pts_1 = []
    pts_2 = []

    # for match in matches:
    #     pts_1.append(pixel2cam(keypoints_1[match.queryIdx].pt, K))
    #     pts_2.append(pixel2cam(keypoints_2[match.trainIdx].pt, K))

    assert len(keypoints_1) == len(keypoints_2)
    for i in range(len(keypoints_1)):
        pts_1.append(pixel2cam(keypoints_1[i], K))
        pts_2.append(pixel2cam(keypoints_2[i], K))

    pts_1 = np.array(pts_1, dtype=np.float32)
    pts_2 = np.array(pts_2, dtype=np.float32)

    pts_4d = cv2.triangulatePoints(T1, T2, pts_1.T, pts_2.T)
    pts_4d /= pts_4d[3]  # 归一化

    points = []
    for i in range(pts_4d.shape[1]):
        x = pts_4d[:, i]
        p = (x[0], x[1], x[2])
        points.append(p)

    return points


def restore_keypoints(keypoint, matches, imageshapeo, imageshaper, method=1):
    oh, ow = imageshapeo
    rh, rw = imageshaper
    scale_x = ow / rw
    scale_y = oh / rh
    keypoint_copy = copy.copy(keypoint)
    for i, kp in enumerate(keypoint):

        x_scaled = kp[0] * scale_x
        y_scaled = kp[1] * scale_y
        # x_restored = x_scaled + (ow - rw) / 2
        # y_restored = y_scaled + (oh - rh) / 2

        keypoint_copy[i] = (x_scaled, y_scaled)
    return keypoint_copy


def reproject3d(keypoints_1, keypoints_2, img_1_copy, img_2_copy, matches, K, name='reproject3d',save_path=''):
    if save_path == '':
        pass
    else:
        os.makedirs(save_path,exist_ok=True)

    # 估计运动
    R, t = pose_estimation_2d2d(keypoints_1, keypoints_2, matches, K)
    print('R:')
    print(R)
    print('T:')
    print(t)
    rotation_vector, _ = cv2.Rodrigues(R)
    rotation_vector_deg = np.rad2deg(rotation_vector)
    print('旋转角(角度制):')
    print(rotation_vector_deg)

    # 三角化
    points = triangulation(keypoints_1, keypoints_2, matches, R, t, K)
    # assert len(matches) == len(points)

    for i in range(len(points)):
        pt1_3d_uv = (
            int((points[i][0] / points[i][2]) * K[0, 0] + K[0, 2]),
            int((points[i][1] / points[i][2]) * K[1, 1] + K[1, 2])
        )

        cv2.circle(img_1_copy, tuple(pt1_3d_uv), 10, (0, 255, 0), -1)
        cv2.circle(img_1_copy, tuple(map(int, keypoints_1[i])), 7, (0, 0, 255), -1)
        cv2.line(img_1_copy, pt1_3d_uv, tuple(map(int, keypoints_1[i])), (255, 0, 0), 2)

        pt1_cam = pixel2cam(keypoints_1[i], K)
        pt1_cam_3d = np.array([
            points[i][0] / points[i][2],
            points[i][1] / points[i][2]
        ], dtype=np.float32)

        # print("Point in the first camera frame:", pt1_cam)
        # print("Point projected from 3D:", pt1_cam_3d, ", d =", points[i][2])

        pt2_cam = pixel2cam(keypoints_2[i], K)
        pt2_trans = R @ np.array([[points[i][0]], [points[i][1]], [points[i][2]]]) + t
        pt2_trans /= pt2_trans[2, 0]

        # print("Point in the second camera frame:", pt2_cam)
        # print("Point reprojected from second frame:", pt2_trans.T)
        # print()

        point3d_img2 = np.array([pt2_trans[0, 0], pt2_trans[1, 0], pt2_trans[2, 0]])
        pt1_3d_uv_img2 = (
            (point3d_img2[0] / point3d_img2[2]) * K[0, 0] + K[0, 2],
            (point3d_img2[1] / point3d_img2[2]) * K[1, 1] + K[1, 2]
        )

        cv2.circle(img_2_copy, tuple(map(int, np.array(pt1_3d_uv_img2).tolist())), 10, (0, 255, 0), -1)
        cv2.circle(img_2_copy, tuple(map(int, keypoints_2[i])), 7, (0, 0, 255), -1)
        pt1_trans = tuple(map(int, np.array(pt1_3d_uv_img2).tolist()))
        pt2_cam = tuple(map(int, keypoints_2[i]))
        cv2.line(img_2_copy, pt1_trans, pt2_cam, (255, 0, 0), 2)

    cv2.imwrite(f"{save_path}/{name}_1.jpg", img_1_copy)
    cv2.imwrite(f"{save_path}/{name}_2.jpg", img_2_copy)

# # 读取图像
# # im_path1 = r'G:\point_match\calibrate\test\WIN_20240220_16_51_21_Pro.jpg'
# # im_path2 = r'G:\point_match\calibrate\test\WIN_20240220_16_51_29_Pro.jpg'
#
# # im_path1 = r'G:\point_match\calibrate\WIN_20240222_17_46_19_Pro.jpg'
# # im_path2 = r'G:\point_match\calibrate\WIN_20240222_17_46_35_Pro.jpg'
# im_path1 = r'G:\point_match\calibrate\1.jpg'
# im_path2 = r'G:\point_match\calibrate\3.jpg'
#
# img_1 = cv2.imread(im_path1, cv2.IMREAD_COLOR)
# img_2 = cv2.imread(im_path2, cv2.IMREAD_COLOR)
#
# if img_1 is None or img_2 is None:
#     print("Failed to load images.")
#
# K = np.array([[1.64235539e+03, 0.00000000e+00, 9.28868433e+02],
#               [0.00000000e+00, 1.64232048e+03, 5.14815213e+02],
#               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
# # 特征点匹配
# keypoints_1, keypoints_2, matches = find_feature_matches(img_1, img_2)
# print("find", len(matches), "matched points")
# img_matches = cv2.drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, None,
#                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imwrite("G:/point_match/calibrate/1_match_points.jpg", img_matches)
# # 估计运动
# R, t = pose_estimation_2d2d(keypoints_1, keypoints_2, matches,K)
# print('R:')
# print(R)
# print('T:')
# print(t)
# rotation_vector, _ = cv2.Rodrigues(R)
# rotation_vector_deg = np.rad2deg(rotation_vector)
# print('旋转角(角度制):')
# print(rotation_vector_deg)
#
# # 三角化
# points = triangulation(keypoints_1, keypoints_2, matches, R, t, K)
# assert len(matches) == len(points)
#
# img_1_copy = img_1.copy()
# img_2_copy = img_2.copy()
#
# for i, match in enumerate(matches):
#     # pt1_3d_uv = np.array([
#     #     (points[i][0] / points[i][2]) * K[0, 0] + K[0, 2],
#     #     (points[i][1] / points[i][2]) * K[1, 1] + K[1, 2]
#     # ], dtype=np.float32)
#
#     pt1_3d_uv = (
#         int((points[i][0] / points[i][2]) * K[0, 0] + K[0, 2]),
#         int((points[i][1] / points[i][2]) * K[1, 1] + K[1, 2])
#     )
#
#     cv2.circle(img_1_copy, tuple(pt1_3d_uv), 10, (0, 255, 0), -1)
#     cv2.circle(img_1_copy, tuple(map(int, keypoints_1[matches[i].queryIdx].pt)), 7, (0, 0, 255), -1)
#     cv2.line(img_1_copy, pt1_3d_uv, tuple(map(int, keypoints_1[matches[i].queryIdx].pt)), (255, 0, 0), 2)
#
#     pt1_cam = pixel2cam(keypoints_1[match.queryIdx].pt, K)
#     pt1_cam_3d = np.array([
#         points[i][0] / points[i][2],
#         points[i][1] / points[i][2]
#     ], dtype=np.float32)
#
#     print("Point in the first camera frame:", pt1_cam)
#     print("Point projected from 3D:", pt1_cam_3d, ", d =", points[i][2])
#
#     pt2_cam = pixel2cam(keypoints_2[match.trainIdx].pt, K)
#     pt2_trans = R @ np.array([[points[i][0]], [points[i][1]], [points[i][2]]]) + t
#     pt2_trans /= pt2_trans[2, 0]
#
#     print("Point in the second camera frame:", pt2_cam)
#     print("Point reprojected from second frame:", pt2_trans.T)
#     print()
#
#
#     point3d_img2 = np.array([pt2_trans[0, 0], pt2_trans[1, 0], pt2_trans[2, 0]])
#     pt1_3d_uv_img2 = (
#         (point3d_img2[0] / point3d_img2[2]) * K[0, 0] + K[0, 2],
#         (point3d_img2[1] / point3d_img2[2]) * K[1, 1] + K[1, 2]
#     )
#
#
#     # cv2.circle(img_2_copy, tuple(pt1_3d_uv_img2), 10, (0, 255, 0), -1)
#     # cv2.circle(img_2_copy, tuple(keypoints_2[match.trainIdx].pt), 7, (0, 0, 255), -1)
#
#     # cv2.circle(img_2_copy, tuple(pt1_3d_uv_img2), 10, (0, 255, 0), -1)
#     cv2.circle(img_2_copy, tuple(map(int,np.array(pt1_3d_uv_img2).tolist())), 10, (0, 255, 0), -1)
#
#     cv2.circle(img_2_copy, tuple(map(int, keypoints_2[match.trainIdx].pt)), 7, (0, 0, 255), -1)
#
#     pt1_trans = tuple(map(int,np.array(pt1_3d_uv_img2).tolist()))
#     pt2_cam = tuple(map(int, keypoints_2[match.trainIdx].pt))
#     cv2.line(img_2_copy, pt1_trans, pt2_cam, (255, 0, 0), 2)
#     # cv2.line(img_2_copy, tuple(pt1_3d_uv_img2), tuple(keypoints_2[match.trainIdx].pt), (255, 0, 0), 2)
#
# cv2.imwrite("G:/point_match/calibrate/1_match.jpg", img_1_copy)
# cv2.imwrite("G:/point_match/calibrate/3_match.jpg", img_2_copy)
#
