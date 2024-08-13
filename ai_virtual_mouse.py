import os
import numpy as np
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm
import autopy

plocX, plocY = 0, 0
clocX, clocY = 0, 0
wCam, hCam = 640, 480
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.HandDetector()
wSrc, hSrc = autopy.screen.size()
frameR = 10
smoothening = 7
while True:
    #1.landmark find
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bbox = detector.findPositions(img)
    #2.tip index and middle
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

    #3.check which fingers ar eup
    fingers = detector.fingersUp()
    #print(fingers)
    cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR), (255, 0, 255), 2)

    #4.index finger:mouse, both index and middle:click
    if fingers[1] == 1 and fingers[2] == 0:
        #convert coordinates
        x3 = np.interp(x1, (frameR, wCam - frameR), (0, wSrc))
        y3 = np.interp(y1, (frameR, hCam - frameR), (0, hSrc))

        #smoothen
        clocX = plocX + (x3 - plocX) / smoothening
        clocY = plocY + (y3 - plocY) / smoothening

        #7.move mouse
        autopy.mouse.move(wSrc - clocX, clocY)
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        plocX, plocY = clocX, clocY
    #8.check if clicking mode
    if fingers[1] == 1 and fingers[2] == 1:
        # find distance between finger
        length, img, lineInfo = detector.findDistance(8, 12, img)
        #print(length)
        # 10 click mouse if distance short

        if length < 35:
            autopy.mouse.click()
            cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 255, 0), cv2.FILLED)

    #fram rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    #display
    cv2.imshow('img', img)
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
