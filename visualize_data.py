#!/usr/bin/env python3

import numpy as np
import cv2
import argparse

def _label2color(label):
    if label == 0:
        return (255, 0, 255)
    elif label == 1:
        return (100, 116, 226)
    elif label == 2:
        return (226, 111, 101)
    elif label == 3:
        return (116, 114, 117)
    elif label == 4:
        return (216, 171, 15)
    else:
        raise('invalid label')


def _draw_bounding_boxes(img, boxes, labels):
    for box, label in zip(boxes, labels):
        img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), _label2color(label), thickness=2)
    return img


def visualize_data_frame(data_file):
    data = np.load(data_file)

    img_rgb = data["arr_0"]
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    boxes = data["arr_1"]
    labels = data["arr_2"]

    print(labels)
    print(boxes)

    return _draw_bounding_boxes(img_bgr, boxes, labels)    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Display image and bounding boxes')

    data_file = 'eval/dataset/48.npz'

    visualization = visualize_data_frame(data_file)

    cv2.imshow('img', visualization)
    cv2.waitKey(0)