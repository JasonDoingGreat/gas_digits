# coding=utf-8

import cv2
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = 'nums_only/'
DIGIT_FILE_NAME = 'digits_only/'

TOTAL_NUM = 0


def detect_digit(img):



def split_image(image, cut_pos):
    global TOTAL_NUM
    for i in range(len(cut_pos)-1):
        new_image = image[:, cut_pos[i]:cut_pos[i+1]]
        new_image = new_image[:, :, 2]
        cv2.imwrite(DIGIT_FILE_NAME + str(TOTAL_NUM) + '.jpg', new_image)
        TOTAL_NUM += 1
    return


def vertical_projection(img):
    height, width, _ = img.shape
    high = height * 2 / 3
    img = img[:, :, 2]

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 13, 1)
    img = cv2.medianBlur(img, 13)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    kernel = np.ones((3, 3), np.uint8)
    img_dilate = cv2.dilate(img, kernel, iterations=2)
    img_erode = cv2.erode(img_dilate, kernel, iterations=4)
    img = cv2.dilate(img_erode, kernel, iterations=2)

    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 对每一列计算投影值
    w = [0] * img.shape[1]
    for x in range(width):
        for y in range(height):
            w[x] += float(img[y, x])/255.
        if int(w[x]) <= 20 or int(w[x]) > high:
            w[x] = 0

    return w


def get_split_position(img):
    result = vertical_projection(img)

    plt.bar(range(len(result)), result)
    plt.show()

    height, width, channel = img.shape
    right_edge = []
    left_edge = []
    for i in range(len(result)-1):
        # 下降边缘
        if result[i] > 0 and result[i+1] == 0:
            right_edge.append(i)
        # 上升边缘
        if result[i] == 0 and result[i+1] > 0:
            if len(left_edge) == 0:
                left_edge.append(i)
                continue
            left_edge.append(i)

    # print "left edge is:", left_edge
    # print "right edge is:", right_edge

    left_edge.pop(0)
    right_edge.pop()

    print right_edge
    print left_edge

    new_left_edge = []
    new_right_edge = []

    for i in range(len(left_edge)):
        if left_edge[i] - right_edge[i] > 10:
            new_left_edge.append(left_edge[i])
            new_right_edge.append(right_edge[i])

    edges = [new_right_edge, new_left_edge]

    cut = np.mean(edges, axis=0)
    cut = [int(i) for i in cut]
    cut.insert(0, 0)
    cut.insert(len(cut), width)

    print "old cut is:", cut

    num_width = []

    for i in range(len(cut) - 1):
        num_width.append(cut[i+1] - cut[i])

    print "width vector is:", num_width

    normal_width = width / 6
    normal_up = normal_width + 20
    normal_down = normal_width - 20

    j = 0
    for i in range(len(num_width)):
        if normal_down <= num_width[i] <= normal_up:
            j += 1
            continue
        else:
            cut_num = int(round(float(num_width[i]) / float(normal_width))) - 1
            increase_width = num_width[i] / (cut_num + 1)
            for k in range(cut_num):
                cut.insert(j+1, cut[j]+increase_width)
                j += 1
            j += 1

    print "new cut is:", cut, '\n'

    # plt.bar(range(len(result)), result)
    # plt.show()
    return cut


def main():
    # for i in range(7):
    #     print "image", i+1
    #     image = FILE_NAME + str(i+1) + '.png'
    #
    #     img = cv2.imread(image)
    #
    #     cut_pos = get_split_position(img)
    #     # split_image(img, cut_pos)

    for 


if __name__ == '__main__':
    main()
