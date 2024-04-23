import subprocess
import os
import cv2
import numpy as np
import math
from glob import glob


def inner_matrix_resize(cameraMatrix, ori_image_shape, resize_image_shape):
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


# 定义处理函数，接受参数并执行处理操作
def process_file(file_path, parameter1, parameter2):
    # 在这里执行文件处理操作，使用不同的参数进行处理
    command = ["python", file_path, '--input', parameter1, '--save_dir', parameter2,
               ]
    subprocess.run(command, check=True)  # 运行命令


if __name__ == '__main__':
    data_path = r'G:\point_match\calibrate\camera_test_gt_val\test_30'
    psave_dir = os.path.join(data_path, 'spsg_process_save_dir')
    os.makedirs(psave_dir, exist_ok=True)

    K_20 = np.array( [[1.50654105e+03, 0.00000000e+00, 9.52765666e+02],
                        [0.00000000e+00, 1.50409791e+03, 5.27824348e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_20 = np.array([[3.59716101e-02, -3.86153710e-01, 8.50330946e-05, -4.09732074e-04, 1.29951151e-01]])

    K_30 = np.array([[1.35121009e+03, 0.00000000e+00, 1.95377803e+03],
                     [0.00000000e+00, 1.34432810e+03, 1.13253609e+03],
                     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    dist_30 = np.array([[-0.01668452, -0.01945304, -0.00125963, -0.00154738,  0.00329328]])

    K_resize = inner_matrix_resize(K_30, (2160, 3840), (1080, 1920))
    fx, fy, cx, cy = K_resize[0, 0], K_resize[1, 1], K_resize[0, 2], K_resize[1, 2]

    # K_30_resize = inner_matrix_resize(K_30, (2160, 3840), (480, 640))
    # fx,fy,cx,cy = K_30_resize[0,0],K_30_resize[1,1],K_30_resize[0,2],K_30_resize[1,2] # 30需要传入
    files = ['1_2','2_3','3_4','4_1'] # ,
    angle = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for file in files:
        data_path_file = os.path.join(data_path,file)
        for angle_i in angle:
            file_path = os.path.join(data_path_file, str(angle_i))
            for item in os.listdir(file_path):
                impath = os.path.join(file_path, item)
                print(f'{angle_i},{item}')
                externs = ['png', 'jpg', 'JPEG', 'BMP', 'bmp']
                image_files = list()
                for extern in externs:
                    image_files.extend(glob(impath + "\\*." + extern))
                assert len(image_files) == 2
                print(f'{image_files[0]}-{image_files[1]},GT:[{item}]')
                process_file(r"G:\point_match\transformer_base\SuperGluePretrainedNetwork\demo_vo2_yongyuyanzheng20_30_zhidingROI.py", impath, impath)


