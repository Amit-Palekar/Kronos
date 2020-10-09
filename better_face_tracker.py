import cv2
import numpy as np

modelFile = "models/res10_300x300_ssd_iter_140000.caffemodel"
configFile = "models/deploy.prototxt.txt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Camera could not be opened.")

DELTA_FRAMES = 0.001;
FRAME_COUNT = 0
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
kf.measurementNoiseCov = cv2.setIdentity(kf.measurementNoiseCov, 1e-7)
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
    frame = cv2.flip(frame, 1)
    # new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # new_frame = cv2.equalizeHist(frame)
    new_frame = cv2.GaussianBlur(frame, (7, 7), 2)


    h, w = new_frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(new_frame, (300, 300)), 1.0,
    (300, 300), (104.0, 117.0, 123.0))
    net.setInput(blob)
    faces = net.forward()
    if FRAME_COUNT % 5 == 0:
        for i in range(faces.shape[2]):
            confidence = faces[0, 0, i, 2]
            if confidence > 0.5:
                box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                frame = cv2.rectangle(frame, (x, y), (x1, y1), (0, 0, 255), 2)
                measurement = np.array([(x1 + x) // 2, (y1 + y) // 2, (x1 + x) // 2 - lastX, (y1 + y) // 2 - lastY], np.float32)
                print("Measurement", measurement, "Confidence", confidence)
                kf.correct(measurement)
                lastX = (x1 + x) // 2
                lastY = (y1 + y) // 2
    prediction = kf.predict()
    center = (prediction[0], prediction[1])
    frame = cv2.ellipse(frame, center, (200, 200), 0, 0, 360, (255, 0, 255), 8)
    FRAME_COUNT += 1
    cv2.imshow('dnn', frame)
    cv2.waitKey(10)
