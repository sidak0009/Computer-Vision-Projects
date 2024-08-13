import cv2
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(self, mode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):
        self.mode = mode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(static_image_mode=self.mode,
                                                 max_num_faces=self.maxFaces,
                                                 min_detection_confidence=self.minDetectionCon,
                                                 min_tracking_confidence=self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)  # Green connections

    def findFaceMesh(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        faces = []

        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_TESSELATION,
                                               landmark_drawing_spec=self.drawSpec,
                                               connection_drawing_spec=self.drawSpec)
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    #cv2.putText(img, str(id), (x, y),
                                #cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

                    face.append([x, y])
                faces.append(face)
        return img, faces


def main():
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector()
    pTime = 0
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if len(faces) != 0:
            print(faces)
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        cv2.imshow('img', img)
        if cv2.waitKey(1) == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    main()
