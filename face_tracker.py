import cv2
import numpy as np

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera could not be opened.")

DELTA_FRAMES = 2.5;
FRAME_COUNT = 10

face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

tMatrix = np.array([[1, 0, DELTA_FRAMES, 0],
                    [0, 1, 0, DELTA_FRAMES],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]], np.float32)

kf = cv2.KalmanFilter(4, 4)
kf.transitionMatrix = tMatrix
kf.measurementMatrix = np.array([[1, 0, DELTA_FRAMES, 0],
                        [0, 1, 0, DELTA_FRAMES],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]], np.float32)
kf.processNoiseCov = cv2.setIdentity(kf.processNoiseCov, 1e-2)
kf.measurementNoiseCov = cv2.setIdentity(kf.measurementNoiseCov, 1e-20)
kf.errorCovPost = cv2.setIdentity(kf.errorCovPost, 1e-1)

print(kf.statePre)
print(kf.transitionMatrix)
print(kf.measurementMatrix)
print(kf.processNoiseCov)
print(kf.measurementNoiseCov)
print(kf.errorCovPost)

lastX = 0
lastY = 0
prediction = kf.predict()
while True:
    ret, frame = cam.read()
    if not ret:
        exit(0)
    frame = cv2.flip(frame, 1)
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    new_frame = cv2.equalizeHist(new_frame)
    new_frame = cv2.GaussianBlur(new_frame, (7, 7), 2)
    faces = face_cascade.detectMultiScale(new_frame, 1.1, 5)
    if len(faces) >= 1:
        x = faces[0][0]+faces[0][2]//2
        y = faces[0][1] + faces[0][3]//2
        measurement = np.array([x, y, x - lastX, y - lastY], np.float32)
        print("Measurement", measurement)
        kf.correct(measurement)
        lastX = x
        lastY = y
        frame = cv2.ellipse(frame, (x, y), (faces[0][2], faces[0][3]), 0, 0, 360, (0, 0, 255), 8)
    if FRAME_COUNT % 10 == 0:
        prediction = kf.predict()
    center = (prediction[0], prediction[1])
    frame = cv2.ellipse(frame, center, (150, 150), 0, 0, 360, (255, 0, 255), 8)
    FRAME_COUNT -= 1
    cv2.imshow('ft', frame)
    # cv2.imshow('ft2', new_frame)
    cv2.waitKey(1)
