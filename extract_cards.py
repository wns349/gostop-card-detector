import os

import cv2
import numpy as np


def _main(args):
    img = cv2.imread(args["img"])  # read image
    img_h, img_w, _ = img.shape
    img_name = os.path.splitext(os.path.basename(args["img"]))[0]

    # using red color as threshold, make white-black image
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])
    mask = cv2.inRange(img_hsv, lower_red, upper_red)
    output_img = img.copy()
    output_img[np.where(mask == 0)] = 0
    output_img[np.where(mask != 0)] = 255

    # find contours
    img_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
    _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    padding = -5  # to remove background
    idx = 1
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if h < 100 or w < 80:  # roi is too small to be true
            continue

        # crop roi
        roi_y_min = max(y - padding, 0)
        roi_y_max = min(y + h + padding, img_h)
        roi_x_min = max(x - padding, 0)
        roi_x_max = min(x + w + padding, img_w)
        roi = img[roi_y_min:roi_y_max, roi_x_min:roi_x_max, :]
        out_name = os.path.join(args["out_dir"], "{}_{}.jpg".format(img_name, idx))
        cv2.imwrite(out_name, roi)
        idx += 1
        print("ROI found and saved at {}".format(out_name))
    print("Done")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img", dest="img", help="Path to source image")
    parser.add_argument("--out_dir", dest="out_dir", help="Path to output directory", default="./")
    _main(vars(parser.parse_args()))
