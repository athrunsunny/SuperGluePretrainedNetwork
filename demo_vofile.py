import os
from pathlib import Path
import argparse
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
from glob import glob
from models.matching import Matching
from models.utils import (AverageTimer, VideoStreamer,
                          make_matching_plot_fast, frame2tensor)

torch.set_grad_enabled(False)
from base_func import pose_estimation_2d2d, pixel2cam, triangulation, reproject3d, restore_keypoints, \
    inner_matrix_resize, get_match


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
    # undistorted_image = cv2.undistort(colorim, K, distort)
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


def parse_opt(known=False):
    parser = argparse.ArgumentParser(
        description='SuperGlue demo',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--input', type=str, default=r'G:\point_match\calibrate\gt\20240305',
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
    parser.add_argument(
        '--method', default='spsg',
        help='Force pytorch to run in CPU mode.')

    return parser.parse_known_args()[0] if known else parser.parse_args()


def main(opt):
    data_path = opt.input
    method = opt.method
    psave_dir = os.path.join(data_path, f'process_save_dir_dlmethod_{method}')
    os.makedirs(psave_dir, exist_ok=True)

    K = np.array([[1.64235539e+03, 0.00000000e+00, 9.28868433e+02],
                  [0.00000000e+00, 1.64232048e+03, 5.14815213e+02],
                  [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

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
    angle = [0, 10, 20, 30, 35, 40, 45]

    for file in os.listdir(data_path):
        image_path = os.path.join(data_path, file)

        externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
        image_files = list()
        for extern in externs:
            image_files.extend(glob(image_path + "\\*." + extern))

        im_path1 = image_files[0]
        frame, cim, ori = load_image(im_path1,opt.resize)
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
            save_dir = os.path.join(psave_dir, file, str(angle[idx]))
            os.makedirs(save_dir, exist_ok=True)

            im_path2 = image
            frame, cim, ori = load_image(im_path2, opt.resize)
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

            K_resize = inner_matrix_resize(K, last_frame_ori, last_frame_color)

            print('-' * 80)
            print('========[resized image]:')
            last_frame_color_copy = last_frame_color.copy()
            cim_copy = cim.copy()
            # pt1_uvs, pt2_uvs, pt1_uv_matchs, pt2_uv_matchs = \
            #     reproject3d(mkpts0, mkpts1, last_frame_color_copy, cim_copy, matches, K_resize,
            #                 name='reproject3d_re', save_path=save_dir)
            get_match(last_frame_color_copy, cim_copy, K_resize, save_dir,
                      keypoints1=mkpts0, keypoints2=mkpts1, match_name='match_points_resize',
                      name='reproject3d_resize_sp', refine=False)

            out = make_matching_plot_fast(
                last_frame_color, cim, kpts0, kpts1, mkpts0, mkpts1, color, text,
                path=None, show_keypoints=opt.show_keypoints, small_text=small_text, margin=0)
            cv2.imwrite(f"{save_dir}/resized_match_points_sp.jpg", out)


if __name__ == '__main__':
    opt = parse_opt(True)
    print(opt)

    main(opt)

