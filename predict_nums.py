
import cv2
import numpy as np
import os
from sklearn.externals import joblib


width = 20
height = 30


def predict_num(img_name):
    clf = joblib.load("trainedValue.pkl")
    training_image = cv2.imread(img_name)
    if training_image is None:
        print "\n\n File doesn't exist \n\n"
        os.system("pause")
        return
    training_image = training_image[:, :, 2]
    resized_image = cv2.resize(training_image, (width, height))
    to_predict = resized_image.reshape((1, height * width))
    to_predict = np.float32(to_predict)

    nbr = clf.predict(to_predict)

    output = str(int(nbr))
    return output, training_image


def main():
    digits = 0

    img_list = os.listdir('digits_only/')
    for img in img_list:
        output, training_image = predict_num('digits_only/'+img)
        print "predict value is:", output
        cv2.imshow("image", training_image)
        cv2.waitKey(0)
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    main()
