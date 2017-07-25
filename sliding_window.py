import imutils
import cv2
import time


def pyramid(image, scale=1.5, min_size=(30, 30)):
    yield image

    while True:
        w = int(image.shape[1] / scale)
        image = imutils.resize(image, width=w)

        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        yield image


def sliding_window(image, step_size, window_size):
    for y in xrange(0, image.shape[0], step_size):
        for x in xrange(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def main():
    image = cv2.imread("images/1.jpg")
    image = cv2.resize(image, (600, 600))[0:300, :]
    (win_height, win_width) = (40, 200)

    for resized in pyramid(image, scale=1.5):
        for (x, y, window) in sliding_window(resized, step_size=16, window_size=(win_width, win_height)):
            if window.shape[0] != win_height or window.shape[1] != win_width:
                continue

            clone = resized.copy()
            cv2.rectangle(clone, (x, y), (x + win_width, y + win_height), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            # time.sleep(0.01)

if __name__ == '__main__':
    main()
