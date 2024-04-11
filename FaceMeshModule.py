import cv2
import mediapipe as mp
import time

class faceMesh():
    def __init__(self):
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh()
        self.mpDraw = mp.solutions.drawing_utils
        
    def findFaceMesh(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS)
        return img
    
    def findPosition(self, img, draw=True):
        lmList = []
        if self.results.multi_face_landmarks:
            for id, lm in enumerate(self.results.multi_face_landmarks[0].landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList
    
def main():
    cap = cv2.VideoCapture(0)
    pTime = 0
    detector = faceMesh()
    while True:
        success, img = cap.read()
        img = detector.findFaceMesh(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        
if __name__ == "__main__":
    main()