# coding=utf-8

import cv2
import numpy as np
import imutils
import os


def pos_gas(img):
    """
    计算获取ID-CARD中ID的位置
    """
    img_copy = img.copy()
    img = cv2.resize(img, (600, 600))
    img_h, img_w, _ = img.shape
    img = img[0:int(0.5 * img_h), 0:img_w]

    # 图像灰度化
    img_gray = img[:, :, 2]

    # 图像平滑
    img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Canny边缘检测
    edged = cv2.Canny(img_gray, 10, 255)
    ret, binary = cv2.threshold(edged, 10, 255, cv2.THRESH_BINARY)

    cv2.imshow("image", binary)
    cv2.waitKey(0)

    # 轮廓检测
    cnts = cv2.findContours(binary.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    print cnts

    # num_pos = [[cv2.boundingRect(c)[1], cv2.boundingRect(c)[0], cv2.boundingRect(c)[2], cv2.boundingRect(c)[3]]
    #            for c in cnts if 13 < cv2.boundingRect(c)[2] < 50 and 25 < cv2.boundingRect(c)[3] < 40]
    #
    # # 将检测出来的位置转化为np格式
    # num_pos = np.array(num_pos)
    # # 合并相同的y坐标
    # pos_y = set(num_pos[:, 0])
    # # 计算y的均值
    # avenge_y = sum(pos_y) / len(pos_y)
    # # 统计大于均值的坐标并计数
    # pos_max = [pos for pos in pos_y if pos > avenge_y + 15]
    # count_max = len(pos_max)
    #
    # # 获取最后的y坐标和最后的坐标
    # final_pos_y = pos_max if 2 * count_max > len(pos_y) else list(set(pos_y).difference(set(pos_max)))
    #
    # final_num_pos = [pos for pos in num_pos if pos[0] in final_pos_y]
    #
    # # 获取坐标的边界
    # final_num_pos = np.array(final_num_pos)
    # final_num_pos[:, 0] = min(final_pos_y)
    #
    # xmax = np.max(final_num_pos[:, 1])
    # xmin = np.min(final_num_pos[:, 1])
    # x_w = np.max(final_num_pos[:, 2])
    # y_h = w = np.max(final_num_pos[:, 3])
    # ymin = np.min(final_num_pos[:, 0])
    #
    # xmax = xmax + x_w
    # ymax = ymin + y_h + 10
    #
    # ymin = ymin + int(0.65 * img_h)
    # ymax = ymax + int(0.65 * img_h)
    #
    # return img_copy, xmax, xmin, ymax, ymin


if __name__ == '__main__':
    img_list = os.listdir("images/")
    for image in img_list:
        print image
        img = cv2.imread("images/" + image)
        pos_gas(img)
        # img_copy, xmax, xmin, ymax, ymin = pos_gas(img)
