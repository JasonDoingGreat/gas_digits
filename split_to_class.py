from sklearn.cluster import KMeans
import scipy.cluster.hierarchy as sch
import numpy as np
import os
import cv2
import matplotlib.pylab as plt

width = 20
height = 30


def readin_samples(path):
    X = []
    img_list = os.listdir(path)
    for img in img_list:
        image = cv2.imread(path+img)
        image = cv2.resize(image, (width, height))

        sobelx = cv2.Sobel(image, cv2.CV_16S, dx=1, dy=0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_16S, dx=0, dy=1, ksize=3)

        absX = cv2.convertScaleAbs(sobelx)
        absY = cv2.convertScaleAbs(sobely)

        dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        cv2.imshow("image", dst)
        cv2.waitKey(0)

        # X.append(image)
    return X, img_list


def main():
    X, img_list = readin_samples('digits_only/')
    # for item in X:
    #     print item
    # estimator = KMeans(init='k-means++', n_clusters=10, n_init=100)
    # estimator.fit(X)
    # print estimator.labels_
    # print img_list

    # disMat = sch.distance.pdist(X, 'euclidean')
    # Z = sch.linkage(disMat, method='average')
    # P = sch.dendrogram(Z)
    #
    # cluster = sch.fcluster(Z, t=1, criterion='inconsistent')
    # print "Original cluster by hierarchy clustering:\n", cluster


if __name__ == '__main__':
    main()
