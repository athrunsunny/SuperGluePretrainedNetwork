import cv2

drawing = False
top_left_pt, bottom_right_pt = (-1, -1), (-1, -1)
callback_finished = False

# 鼠标回调函数
def mouse_callback(event, x, y, flags, param):
    global  top_left_pt, bottom_right_pt, drawing,callback_finished

    image,save_path,count = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        top_left_pt = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bottom_right_pt = (x, y)
        # cv2.rectangle(image, top_left_pt, bottom_right_pt, (0, 255, 0), 1)
        # cv2.imshow("Image", image)
        # 裁剪选定的区域
        cropped_image = image[int(top_left_pt[1]):int(bottom_right_pt[1]), int(top_left_pt[0]):int(bottom_right_pt[0])]
        # cv2.imshow("Cropped Image", cropped_image)
        # cv2.waitKey(0)
        cv2.imwrite('%s/%s_%s.jpg'%(save_path,'cropped_image',str(count)), cropped_image)
        print('%s/%s_%s.jpg'%(save_path,'cropped_image',str(count)))
        # print(cropped_image)
        print("裁剪后的图像已保存为 cropped_image.jpg")

        with open('%s/%s_%s.txt'%(save_path,'cropped_image',str(count)), 'w') as f:
            f.write(f"{top_left_pt[0]}, {top_left_pt[1]},{bottom_right_pt[0]}, {bottom_right_pt[1]}")
            print("起始位置已保存到 start_position.txt")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # 设置回调函数完成的标志
        callback_finished = True
        return
    if drawing:
        image_copy = image.copy()
        cv2.rectangle(image_copy, top_left_pt, (x, y), (0, 255, 0), 1)
        cv2.putText(image_copy, f"Width: {abs(x - top_left_pt[0])}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0),
                    2)
        cv2.putText(image_copy, f"Height: {abs(y - top_left_pt[1])}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
        cv2.imshow("Image", image_copy)


# # 加载图像
# image = cv2.imread(r'G:\point_match\calibrate\camera_test_gt_val\20240415\30_20_zibiaoding30_test\8fa1079d60c44b29894638bb407f06e8.png')
#
# # 创建窗口并绑定鼠标回调函数
# cv2.namedWindow("Image")
# cv2.setMouseCallback("Image", mouse_callback)
#

#
# # 显示图像
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
