import cv2
import mediapipe as mp
import time

cap=cv2.VideoCapture(0)

wCam, hCam = 640, 480
pTime = 0
cap.set(3, wCam)
cap.set(4, hCam)

while True:
    success, img = cap.read()
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS:{int(fps)}', (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
    cv2.imshow('img', img)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break