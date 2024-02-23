import cv2
import numpy as np
from matplotlib import pyplot as plt

from models.utils import make_matching_plot_fast

cap = cv2.VideoCapture(r'G:\point_match\WIN_20240202_09_06_05_Pro.mp4')  # 替换为你的视频文件路径
ret, frame1 = cap.read()
frame1 = cv2.resize(frame1, (640, 480))
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)

output_video = cv2.VideoWriter(r'G:\point_match\transformer_base\SuperGluePretrainedNetwork\sift.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 480))
frame_id = 0
while True:
    ret, frame2 = cap.read()
    if not ret:
        break
    frame2 = cv2.resize(frame2, (640, 480))
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    kp2, des2 = sift.detectAndCompute(gray2, None)

    # 使用FLANN匹配器进行特征点匹配
    flann = cv2.FlannBasedMatcher()
    matches = flann.knnMatch(des1, des2, k=2)

    # 选择良好的匹配
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 可视化匹配的特征点

    text = [
        'SuperGlue',
        'Keypoints: {}:{}'.format(len(kp1), len(kp2)),
        'Matches: {}'.format(len(good_matches))
    ]
    k_thresh = 0.0050
    m_thresh = 0.20
    small_text = [
        'Keypoint Threshold: {:.4f}'.format(k_thresh),
        'Match Threshold: {:.2f}'.format(m_thresh),
        'Image Pair: {:06}:{:06}'.format(0, frame_id),
    ]
    # out = make_matching_plot_fast(
    #     frame1, frame2, kp1, kp2, mkpts0, mkpts1, color, text,
    #     path=None, show_keypoints=False, small_text=small_text)
    img_matches = cv2.drawMatches(gray1, kp1, gray2, kp2, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # g1 = img_matches[:,:640]
    # g2 = img_matches[:,640:]
    margin = 0
    H0, W0 = gray1.shape[:2]
    H1, W1 = gray2.shape[:2]
    H, W = max(H0, H1), W0 + W1 + margin
    sc = min(H / 640., 2.0)
    Ht = int(30 * sc)  # text height

    # out = 255 * np.ones((H, W), np.uint8)
    # out[:H0, :W0] = gray1
    # out[:H1, W0 + margin:] = gray2
    # out = np.stack([out] * 3, -1)
    out = img_matches
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), Ht * (i + 1)), cv2.FONT_HERSHEY_DUPLEX,
                    1.0 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # Small text.
    Ht = int(18 * sc)  # text height
    for i, t in enumerate(reversed(small_text)):
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (int(8 * sc), int(H - Ht * (i + .6))), cv2.FONT_HERSHEY_DUPLEX,
                    0.5 * sc, txt_color_fg, 1, cv2.LINE_AA)

    # img_matches = cv2.drawMatches(frame1, kp1, frame2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    output_video.write(img_matches)
    cv2.imshow('Matches', img_matches)
    frame_id += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
