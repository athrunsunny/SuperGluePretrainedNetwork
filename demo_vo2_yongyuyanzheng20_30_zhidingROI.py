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
from glob import glob

from draw_rectangle import mouse_callback
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)
from base_func import pose_estimation_2d2d, pixel2cam, triangulation, reproject3d, restore_keypoints, \
    inner_matrix_resize, get_match


def inner_matrix_resize_shape(cameraMatrix, ori_image_shape, resize_image_shape):
    oh, ow = ori_image_shape
    rh, rw = resize_image_shape
    scale_x = rw / ow
    scale_y = rh / oh

    k00 = cameraMatrix[0, 0] * scale_x
    k11 = cameraMatrix[1, 1] * scale_y
    k02 = cameraMatrix[0, 2] * scale_x
    k12 = cameraMatrix[1, 2] * scale_y
    cameraMatrix_resize = np.array([[k00, 0.0, k02], [0.0, k11, k12], [0.0, 0.0, 1.0]])
    return cameraMatrix_resize


def process_resize(w, h, resize):
    assert (len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def crop(impath, resize, interp=cv2.INTER_AREA, save_path='', count=0):
    """ Read image as grayscale and resize to img_size.
    Inputs
        impath: Path to input image.
    Returns
        grayim: uint8 numpy array sized H x W.
    """
    colorim = cv2.imread(impath)

    # 去畸变
    K_20 = np.array([[1.64235539e+03, 0.00000000e+00, 9.28868433e+02],
                     [0.00000000e+00, 1.64232048e+03, 5.14815213e+02],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_20 = np.array([0.13630707, -1.032316, -0.001774, -0.00442506, 1.71583531])

    # K_20 = np.array([[1.50654105e+03, 0.00000000e+00, 9.52765666e+02],
    #                  [0.00000000e+00, 1.50409791e+03, 5.27824348e+02],
    #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist_20 = np.array([[3.59716101e-02, -3.86153710e-01, 8.50330946e-05, -4.09732074e-04, 1.29951151e-01]])

    # K_30 = np.array([[1.30382991e+03, 0.00000000e+00, 1.96542499e+03],
    #                  [0.00000000e+00, 1.29883157e+03, 1.11729905e+03],
    #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist_30 = np.array([[4.43930058e-05, -2.85624595e-02, -2.06776952e-03, -1.83244312e-03, 6.85512086e-03]])

    K_30 = np.array([[1.35121009e+03, 0.00000000e+00, 1.95377803e+03],
                    [0.00000000e+00, 1.34432810e+03, 1.13253609e+03],
                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_30 = np.array([[-0.01668452, -0.01945304, -0.00125963, -0.00154738,  0.00329328]])

    undistorted_image = cv2.undistort(colorim, K_30, dist_30)
    colorim = undistorted_image

    undist_image_path = '/'.join(impath.replace('\\', '/').split('/')[:-1])
    cv2.imwrite('%s/%s_%s.jpg' % (save_path, 'undistort_image', str(count)), undistorted_image)

    undistorted_image_copy = undistorted_image.copy()
    undistorted_image_copy = cv2.resize(undistorted_image_copy,
                                        (undistorted_image_copy.shape[1] // 2, undistorted_image_copy.shape[0] // 2))

    cv2.namedWindow("Image")
    cv2.setMouseCallback("Image", mouse_callback, param=(undistorted_image_copy, save_path, count))
    # 显示图像
    cv2.imshow("Image", undistorted_image_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # ori_image = colorim.copy()
    # grayim = colorim
    # # grayim = cv2.imread(impath, 0)
    # if grayim is None:
    #     raise Exception('Error reading image %s' % impath)
    # w, h = grayim.shape[1], grayim.shape[0]
    # w_new, h_new = process_resize(w, h, resize)
    # grayim = cv2.resize(
    #     grayim, (w_new, h_new), interpolation=interp)
    #
    # colorim = grayim
    # grayim = cv2.cvtColor(colorim, cv2.COLOR_BGR2GRAY)
    # return grayim, colorim, ori_image


def load_image(impath, resize, interp=cv2.INTER_AREA):
    """ Read image as grayscale and resize to img_size.
    Inputs
        impath: Path to input image.
    Returns
        grayim: uint8 numpy array sized H x W.
    """
    colorim = cv2.imread(impath)

    # 去畸变
    # K = np.array([[1.64235539e+03, 0.00000000e+00, 9.28868433e+02],
    #               [0.00000000e+00, 1.64232048e+03, 5.14815213e+02],
    #               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # distort = np.array([ 0.13630707, -1.032316,   -0.001774,   -0.00442506,  1.71583531])
    #
    # K_20 = np.array([[1.50654105e+03, 0.00000000e+00, 9.52765666e+02],
    #                  [0.00000000e+00, 1.50409791e+03, 5.27824348e+02],
    #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist_20 = np.array([[3.59716101e-02, -3.86153710e-01, 8.50330946e-05, -4.09732074e-04, 1.29951151e-01]])
    #
    # K_30 = np.array([[1.30382991e+03, 0.00000000e+00, 1.96542499e+03],
    #                  [0.00000000e+00, 1.29883157e+03, 1.11729905e+03],
    #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist_30 = np.array([[4.43930058e-05, -2.85624595e-02, -2.06776952e-03, -1.83244312e-03, 6.85512086e-03]])

    # K_30 = np.array([[1.35121009e+03, 0.00000000e+00, 1.95377803e+03],
    #                 [0.00000000e+00, 1.34432810e+03, 1.13253609e+03],
    #                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # dist_30 = np.array([[-0.01668452, -0.01945304, -0.00125963, -0.00154738,  0.00329328]])
    # #
    # undistorted_image = cv2.undistort(colorim, K_30, dist_30)
    # colorim = undistorted_image

    ori_image = colorim.copy()
    grayim = colorim
    # grayim = cv2.imread(impath, 0)
    if grayim is None:
        raise Exception('Error reading image %s' % impath)
    w, h = grayim.shape[1], grayim.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    grayim = cv2.resize(
        grayim, (w_new, h_new), interpolation=interp)

    colorim = grayim
    grayim = cv2.cvtColor(colorim, cv2.COLOR_BGR2GRAY)
    return grayim, colorim, ori_image

def read_txt(path):
    assert os.path.exists(path)
    with open(path, mode='r', encoding="utf-8") as f:
        content = f.readlines()
    content = np.array(content)
    res = []
    for index, item in enumerate(content):
        string = item.split(',')
        res.append(list(map(np.float64, string)))
    # 防止重叠目标
    res = np.array(res)
    # res = np.unique(res, axis=0)
    # if len(res) == 0:
    #     return None
    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default=r'G:\point_match\calibrate\camera_test_gt_val\20240415\70_0',
        help='ID of a USB webcam, URL of an IP camera, '
             'or path to an image directory or movie file')
    parser.add_argument(
        '--save_dir', type=str, default=r'G:\point_match\calibrate\camera_test_gt_val\20240415\70_0',
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
    parser.add_argument(
        '--original_size', type=int, nargs='+', default=[1920, 1080],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')

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

    data_path = opt.input
    psave_dir = '%s/%s'%(opt.save_dir,'cropped_image')
    os.makedirs(psave_dir,exist_ok=True)
    externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
    image_files = list()
    for extern in externs:
        image_files.extend(glob(data_path + "\\*." + extern))

    im_path1 = image_files[0]
    crop(im_path1, opt.resize, save_path=psave_dir, count=1)
    cropped_image_path_1 = '%s/%s'%(psave_dir,'cropped_image_1.jpg')
    frame, cim, ori = load_image(cropped_image_path_1, opt.resize)

    tlbr1 = read_txt('%s/%s.txt' % (psave_dir,'cropped_image_1'))[0]
    print(tlbr1)
    _, frame_original_size1, _ = load_image(im_path1, opt.original_size)

    frame_tensor = frame2tensor(frame, device)
    last_data = matching.superpoint({'image': frame_tensor})
    last_data = {k + '0': last_data[k] for k in keys}
    last_data['image0'] = frame_tensor
    last_frame = frame
    last_image_id = 0
    last_frame_color = cim
    last_frame_ori = ori

    oh, ow, _ = last_frame_ori.shape
    rh, rw, _ = last_frame_color.shape

    for idx, image in enumerate(image_files[1:]):
        timer = AverageTimer()
        save_dir = psave_dir
        os.makedirs(save_dir, exist_ok=True)

        im_path2 = image
        # frame, cim, ori = load_image(im_path2, opt.resize)

        crop(im_path2, opt.resize, save_path=psave_dir, count=2)
        cropped_image_path_2 = '%s/%s' % (psave_dir, 'cropped_image_2.jpg')
        frame, cim, ori = load_image(cropped_image_path_2, opt.resize)
        tlbr2 = read_txt('%s/%s.txt' % (psave_dir, 'cropped_image_2'))[0]
        print(tlbr2)
        _, frame_original_size2, _ = load_image(im_path2, opt.original_size)

        oh1, ow1, _ = ori.shape
        rh1, rw1, _ = cim.shape

        timer.update('data')

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

        assert len(mkpts0) == len(mkpts1)
        if len(mkpts0) < 8:
            continue

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
        ]

        # K_resize = inner_matrix_resize(K, last_frame_ori, last_frame_color)
        #
        # print('-' * 80)
        # print('========[resized image]:')
        # last_frame_color_copy = last_frame_color.copy()
        # cim_copy = cim.copy()
        # # pt1_uvs, pt2_uvs, pt1_uv_matchs, pt2_uv_matchs = \
        # #     reproject3d(mkpts0, mkpts1, last_frame_color_copy, cim_copy, matches, K_resize,
        # #                 name='reproject3d_re', save_path=save_dir)
        # get_match(last_frame_color_copy, cim_copy, K_resize, save_dir,
        #           keypoints1=mkpts0, keypoints2=mkpts1, match_name='match_points_resize',
        #           name='reproject3d_resize_sp', refine=False)

        out = make_matching_plot_fast(
            last_frame_color, cim, kpts0, kpts1, mkpts0, mkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text, margin=0)
        cv2.imwrite(f"{save_dir}/resized_match_points_sp.jpg", out)

        # get_match(last_frame_color, cim, np.eye(3), save_dir,
        #           keypoints1=kpts0, keypoints2=kpts1, match_name='match_points_resize',
        #           name='reproject3d_resize_sp11', refine=False)

        scale_h1 = rh / oh
        scale_w1 = rw / ow

        scale_h2 = rh1 / oh1
        scale_w2 = rw1 / ow1

        print(oh, ow, oh1, ow1)
        print(rh, rw, rh1, rw1)
        print(scale_w1,scale_h1)
        print(scale_w2,scale_h2)
        # np.save(r'G:\point_match\calibrate\camera_test_gt_val\20240415\30_20_zibiaoding30_test\cropped_image\kp1.npy', mkpts0)
        # np.save(r'G:\point_match\calibrate\camera_test_gt_val\20240415\30_20_zibiaoding30_test\cropped_image\kp2.npy', mkpts1)

        osmkpts0 = mkpts0 / np.array([scale_w1,scale_h1]) + np.array([tlbr1[0],tlbr1[1]])
        osmkpts1 = mkpts1 / np.array([scale_w2,scale_h2]) + np.array([tlbr2[0],tlbr2[1]])

        out = make_matching_plot_fast(
            frame_original_size1, frame_original_size2, osmkpts0, osmkpts1, osmkpts0, osmkpts1, color, text,
            path=None, show_keypoints=opt.show_keypoints, small_text=small_text, margin=0)
        cv2.imwrite(f"{save_dir}/resized_match_points_sp_o.jpg", out)


        # K_30 = np.array([[1.30382991e+03, 0.00000000e+00, 1.96542499e+03],
        #                  [0.00000000e+00, 1.29883157e+03, 1.11729905e+03],
        #                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        K_30 = np.array([[1.35121009e+03, 0.00000000e+00, 1.95377803e+03],
                        [0.00000000e+00, 1.34432810e+03, 1.13253609e+03],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        # dist_30 = np.array([[-0.01668452, -0.01945304, -0.00125963, -0.00154738,  0.00329328]])
        K_resize = inner_matrix_resize_shape(K_30, (2160, 3840), (1080, 1920))
        get_match(frame_original_size1, frame_original_size2, K_resize, save_dir,
                  keypoints1=osmkpts0, keypoints2=osmkpts1, match_name='match_points_resize',
                  name='reproject3d_resize_sp1', refine=False)







