import argparse
import os
import cv2
import numpy as np

args = argparse.ArgumentParser()
args.add_argument('-f', "--folder", required=True, help="Path to images directory")
args = vars(args.parse_args())

folders = os.listdir(args['folder'])
for f in folders:
    images = []
    imagePath = os.listdir(os.path.sep.join([args['folder'], f]))
    imagePath.sort()
    for i in imagePath:
        img = cv2.imread(os.path.sep.join([args['folder'], f, i]))
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        label = i.split('-')[-1].split('.')[0]
        cv2.putText(img, label, (10, 25), font, 1, (255,255,255), 1, cv2.LINE_AA)
        images.append(img)
    numpy_vertical = np.vstack((images[0], images[1]))
    numpy_vertical_concat = np.concatenate((images[0], images[1]), axis=0)
    cv2.imwrite('./prediction/combined/'+f+'.jpg',numpy_vertical_concat)