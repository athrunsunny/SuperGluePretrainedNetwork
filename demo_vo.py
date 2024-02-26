#! /usr/bin/env python3
#
# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%
import os
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np

from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)
from base_func import pose_estimation_2d2d, pixel2cam, triangulation, reproject3d, restore_keypoints

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default=r'G:\point_match\calibrate\45',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--output_dir', type=str, default=r'G:\point_match\transformer_base\SuperGluePretrainedNetwork',
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[640, 480],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

    parser.add_argument(
        '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
        help='SuperGlue weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=-1,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='SuperGlue match threshold')

    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true', default=True,
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    # parser.add_argument(
    #     '--output_path', type=str, default=r'',
    #     help='Force pytorch to run in CPU mode.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    device = 'cuda' if torch.cuda.is_available() and not opt.force_cpu else 'cpu'
    print('Running inference on device \"{}\"'.format(device))
    config = {
        'superpoint': {
            'nms_radius': opt.nms_radius,
            'keypoint_threshold': opt.keypoint_threshold,
            'max_keypoints': opt.max_keypoints
        },
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    matching = Matching(config).eval().to(device)
    keys = ['keypoints', 'scores', 'descriptors']

    vs = VideoStreamer(opt.input, opt.resize, opt.skip,
                       opt.image_glob, opt.max_length)
    (frame, ret), cim, ori = vs.next_frame()
    assert ret, 'Error when reading the first frame (try different --input?)'

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k+'0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0
    last_frame_color = cim
    last_frame_ori = ori

    if opt.output_dir is not None:
        print('==> Will write outputs to {}'.format(opt.output_dir))
        Path(opt.output_dir).mkdir(exist_ok=True)

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', 640*2, 480)
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')

    timer = AverageTimer()
    save_path = opt.output_dir
    fps, w, h = 30, int(opt.resize[0]) * 2, int(opt.resize[1])
    save_path += '/out2.mp4'
    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h), isColor=True)
    while True:
        (frame, ret), cim, ori = vs.next_frame()
        if not ret:
            print('Finished demo_superglue.py')
            break
        timer.update('data')
        stem0, stem1 = last_image_id, vs.i - 1

        frame_tensor = frame2tensor(frame, device)
        pred = matching({**last_data, 'image1': frame_tensor})
        kpts0 = last_data['keypoints0'][0].cpu().numpy()
        kpts1 = pred['keypoints1'][0].cpu().numpy()
        matches = pred['matches0'][0].cpu().numpy()
        confidence = pred['matching_scores0'][0].cpu().numpy()
        timer.update('forward')

        valid = matches > -1
        mkpts0 = kpts0[valid]
        mkpts1 = kpts1[matches[valid]]
        color = cm.jet(confidence[valid])
        text = [
            'SuperGlue',
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]
        k_thresh = matching.superpoint.config['keypoint_threshold']
        m_thresh = matching.superglue.config['match_threshold']
        small_text = [
            'Keypoint Threshold: {:.4f}'.format(k_thresh),
            'Match Threshold: {:.2f}'.format(m_thresh),
            'Image Pair: {:06}:{:06}'.format(stem0, stem1),
        ]
        save_dir = r'G:\point_match\calibrate\tested'
        os.makedirs(save_dir, exist_ok=True)

        points1 = np.float32(mkpts0).reshape(-1, 1, 2)
        points2 = np.float32(mkpts1).reshape(-1, 1, 2)
        # camera_matrix = np.array([[1.64235539e+03, 0.00000000e+00, 9.28868433e+02],
        #       [0.00000000e+00, 1.64232048e+03, 5.14815213e+02],
        #       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        #
        # R, t = pose_estimation_2d2d(mkpts0, mkpts1, matches, camera_matrix)
        #
        # print('R:')
        # print(R)
        # print('T:')
        # print(t)
        # rotation_vector, _ = cv2.Rodrigues(R)
        # rotation_vector_deg = np.rad2deg(rotation_vector)
        # print('旋转角(角度制):')
        # print(rotation_vector_deg)
        #
        # points = triangulation(mkpts0, mkpts1, matches, R, t, camera_matrix)
        #
        # img_1_copy = last_frame_color.copy()
        # img_2_copy = cim.copy()
        # K = camera_matrix
        # for i in range(len(points)):
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
        #     cv2.circle(img_1_copy, tuple(map(int, mkpts0[i])), 7, (0, 0, 255), -1)
        #     cv2.line(img_1_copy, pt1_3d_uv, tuple(map(int, mkpts0[i])), (255, 0, 0), 2)
        #
        #     pt1_cam = pixel2cam(mkpts0[i], K)
        #     pt1_cam_3d = np.array([
        #         points[i][0] / points[i][2],
        #         points[i][1] / points[i][2]
        #     ], dtype=np.float32)
        #
        #     print("Point in the first camera frame:", pt1_cam)
        #     print("Point projected from 3D:", pt1_cam_3d, ", d =", points[i][2])
        #
        #     pt2_cam = pixel2cam(mkpts1[i], K)
        #     pt2_trans = R @ np.array([[points[i][0]], [points[i][1]], [points[i][2]]]) + t
        #     pt2_trans /= pt2_trans[2, 0]
        #
        #     print("Point in the second camera frame:", pt2_cam)
        #     print("Point reprojected from second frame:", pt2_trans.T)
        #     print()
        #
        #     point3d_img2 = np.array([pt2_trans[0, 0], pt2_trans[1, 0], pt2_trans[2, 0]])
        #     pt1_3d_uv_img2 = (
        #         (point3d_img2[0] / point3d_img2[2]) * K[0, 0] + K[0, 2],
        #         (point3d_img2[1] / point3d_img2[2]) * K[1, 1] + K[1, 2]
        #     )
        #
        #     # cv2.circle(img_2_copy, tuple(pt1_3d_uv_img2), 10, (0, 255, 0), -1)
        #     # cv2.circle(img_2_copy, tuple(keypoints_2[match.trainIdx].pt), 7, (0, 0, 255), -1)
        #
        #     # cv2.circle(img_2_copy, tuple(pt1_3d_uv_img2), 10, (0, 255, 0), -1)
        #     cv2.circle(img_2_copy, tuple(map(int, np.array(pt1_3d_uv_img2).tolist())), 10, (0, 255, 0), -1)
        #
        #     cv2.circle(img_2_copy, tuple(map(int, mkpts1[i])), 7, (0, 0, 255), -1)
        #
        #     pt1_trans = tuple(map(int, np.array(pt1_3d_uv_img2).tolist()))
        #     pt2_cam = tuple(map(int, mkpts1[i]))
        #     cv2.line(img_2_copy, pt1_trans, pt2_cam, (255, 0, 0), 2)
        #     # cv2.line(img_2_copy, tuple(pt1_3d_uv_img2), tuple(keypoints_2[match.trainIdx].pt), (255, 0, 0), 2)
        #
        # cv2.imwrite("G:/point_match/calibrate/10_match.jpg", img_1_copy)
        # cv2.imwrite("G:/point_match/calibrate/30_match.jpg", img_2_copy)

        K = np.array([[1.64235539e+03, 0.00000000e+00, 9.28868433e+02],
                      [0.00000000e+00, 1.64232048e+03, 5.14815213e+02],
                      [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        oh, ow, _ = last_frame_ori.shape
        rh, rw, _ = last_frame_color.shape
        scale_x = rw / ow
        scale_y = rh / oh

        k00 = K[0, 0] * scale_x
        k11 = K[1, 1] * scale_y
        k02 = K[0, 2] * scale_x
        k12 = K[1, 2] * scale_y
        K_resize = np.array([[k00, 0.0, k02], [0.0, k11, k12], [0.0, 0.0, 1.0]])

        last_frame_color_copy = last_frame_color.copy()
        cim_copy = cim.copy()
        reproject3d(mkpts0, mkpts1, last_frame_color_copy, cim_copy, matches, K_resize, name='reproject3d_re', save_path=save_dir)
        out = make_matching_plot_fast(
            last_frame_color, cim, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text,margin=0)
        cv2.imwrite(f"{save_dir}/resized_match_points.jpg", out)

        last_frame_ori_copy = last_frame_ori.copy()
        ori_copy = ori.copy()
        keypoints_1o = restore_keypoints(mkpts0, matches, (oh, ow), (rh, rw), method=1)
        keypoints_2o = restore_keypoints(mkpts1, matches, (oh, ow), (rh, rw), method=2)
        reproject3d(keypoints_1o, keypoints_2o, last_frame_ori, ori, matches, K, name='reproject3d_o', save_path=save_dir)

        out = make_matching_plot_fast(
            last_frame_ori_copy, ori_copy, kpts0, kpts1, keypoints_1o, keypoints_2o, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text,margin=0)
        cv2.imwrite(f"{save_dir}/original_match_points.jpg", out)

        if not opt.no_display:
            cv2.imshow('SuperGlue matches', out)
            key = chr(cv2.waitKey(1) & 0xFF)
            if key == 'q':
                vs.cleanup()
                print('Exiting (via q) demo_superglue.py')
                break
            elif key == 'n':  # set the current frame as anchor
                last_data = {k+'0': pred[k+'1'] for k in keys}
                last_data['image0'] = frame_tensor
                last_frame = frame
                last_image_id = (vs.i - 1)
            elif key in ['e', 'r']:
                # Increase/decrease keypoint threshold by 10% each keypress.
                d = 0.1 * (-1 if key == 'e' else 1)
                matching.superpoint.config['keypoint_threshold'] = min(max(
                    0.0001, matching.superpoint.config['keypoint_threshold']*(1+d)), 1)
                print('\nChanged the keypoint threshold to {:.4f}'.format(
                    matching.superpoint.config['keypoint_threshold']))
            elif key in ['d', 'f']:
                # Increase/decrease match threshold by 0.05 each keypress.
                d = 0.05 * (-1 if key == 'd' else 1)
                matching.superglue.config['match_threshold'] = min(max(
                    0.05, matching.superglue.config['match_threshold']+d), .95)
                print('\nChanged the match threshold to {:.2f}'.format(
                    matching.superglue.config['match_threshold']))
            elif key == 'k':
                opt.show_keypoints = not opt.show_keypoints

        timer.update('viz')
        timer.print()

        if opt.output_dir is not None:
            #stem = 'matches_{:06}_{:06}'.format(last_image_id, vs.i-1)
            # stem = 'matches_{:06}_{:06}'.format(stem0, stem1)
            # out_file = str(Path(opt.output_dir, stem + '.png'))
            # print('\nWriting image to {}'.format(out_file))
            # cv2.imwrite(out_file, out)

            vid_writer.write(out)
    cv2.destroyAllWindows()
    vs.cleanup()
