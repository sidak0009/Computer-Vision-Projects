import os
import numpy as np
import cv2
import mediapipe as mp
import time
import HandTrackingModule as htm

brushThickness = 15
xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)
eraserThick = 100

folderPath = "header"
mylist = os.listdir(folderPath)
overlayList = []
for imPath in mylist:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
cap = cv2.VideoCapture(0)
header = overlayList[0]

pTime = 0
cap.set(3, 1280)
cap.set(4, 720)
drawColor = (255, 0, 255)

detector = htm.HandDetector(detectioncon=0.85)
while True:
    success, img = cap.read()
    img[0:127, 0:1280] = header

    img = cv2.flip(img, 1)

    img = detector.findHands(img)
    lmslist = detector.findPositions(img, draw=False)
    if len(lmslist) != 0:

        x1, y1 = lmslist[8][1:]
        x2, y2 = lmslist[12][1:]

    fingers = detector.fingersUp()
    #print(fingers)

    if fingers[1] and fingers[2]:
        #print("selection mode")
        xp, yp = 0, 0

        if y1 < 127:
            if 0 < x1 < 240:
                header = overlayList[3]
                #print("eraser")
                drawColor = (0, 0, 0)

            elif 268 < x1 < 521:
                header = overlayList[2]
                #print("green")
                drawColor = (0, 255, 0)

            elif 561 < x1 < 887:
                header = overlayList[1]
                #print("blue")
                drawColor = (255, 0, 0)

            else:
                header = overlayList[0]
                #print("pink sahi aa")
                drawColor = (255, 0, 255)

        cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)

    if fingers[1] and fingers[2] == False:
        cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
        if xp == 0 and yp == 0:
            xp, yp = x1, y1
        if drawColor == (0, 0, 0):
            cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThick)
            cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThick)
        cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
        cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)

        xp, yp = x1, y1

        #print("draw")
    imGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    ret, imgInv = cv2.threshold(imGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img=cv2.bitwise_and(img,imgInv
                        )
    img=cv2.bitwise_or(img,imgCanvas)
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow('img', img)
    #cv2.imshow('imsg', imgCanvas)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break
