import os
import numpy as np
import cv2
import time
import HandTrackingModule as htm
import autopy
import math

cap = cv2.VideoCapture(0)
detector = htm.HandDetector(maxHands=1)
offset = 30
imgSize = 300
folder = "Data/peace_sign"
counter = 1

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lmlist, bbox = detector.findPositions(img)
    if bbox:
        x = bbox[0]
        y = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h, x - offset:x + w + 25]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (hCal, imgSize))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[:, hGap:hCal + hGap] = imgResize
        cv2.imshow('crop', imgCrop)
        cv2.imshow('white', imgWhite)

    cv2.imshow('img', img)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
    if cv2.waitKey(1) == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image{time.time()}.jpg', imgWhite)
        print(counter)
