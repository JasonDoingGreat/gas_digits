from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import os

img_list = os.listdir("digits_only/")

for img in img_list:
    image = cv2.imread("digits_only/" + img)
    image = image[:, :, 2]
    image = cv2.GaussianBlur(image, 3)
    sobelx = cv2.Sobel(image, cv2.CV_16S, dx=1, dy=0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_16S, dx=0, dy=1, ksize=3)

    absX = cv2.convertScaleAbs(sobelx)
    absY = cv2.convertScaleAbs(sobely)

    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

    cv2.imshow("image", dst)
    cv2.waitKey(0)

