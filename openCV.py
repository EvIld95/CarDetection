import cv2
import numpy as np


def calculateMedian(table):
    sum = 0
    amount = 0
    for x in table:
        for y in x:
            amount += 1
            sum += y
    return sum // amount


cap = cv2.VideoCapture('converted.mov')

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.createBackgroundSubtractorKNN()
# carCascade = cv2.CascadeClassifier('cars.xml')

params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
params.filterByColor = True
counter = 0
detector = cv2.SimpleBlobDetector_create(parameters=params)

while True:
    _, frame = cap.read()
    media = cv2.medianBlur(frame, 3)
    blur = cv2.GaussianBlur(frame, (5, 5), 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask2 = fgbg2.apply(media)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    close = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    # erode = cv2.erode(opening, (10,10))
    # dilation = cv2.dilate(opening, kernel, iterations=3)
    thresh = 200
    binary = (opening > thresh) * 255
    image = np.uint8(binary)

    keypoints = detector.detect(image)

    for keypoint in keypoints:
        _, y = keypoint.pt
        if y > 380 and keypoint.class_id < counter:
            counter = counter + 1
            keypoint.class_id = counter
            print('detected {}'.format(counter))

    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255))

    # treshold = 10
    # for x in range(0, image.shape[0] // treshold):
    #     for y in range(0, image.shape[1] // treshold):
    #         median = calculateMedian(image[treshold * x:(treshold * x) + treshold, treshold * y:(treshold * y) + treshold])
    #
    #
    #         if median > 250:
    #             image[treshold * x:(treshold * x) + treshold, treshold * y:(treshold * y) + treshold] = 255
    #         else:
    #             image[treshold * x:(treshold * x) + treshold, treshold * y:(treshold * y) + treshold] = 0

    # cars = carCascade.detectMultiScale(gray, 1.05, 5)
    # for (x,y,w,h) in cars:
    #     cv2.rectangle(frame, (x,y), (x+w, y+(h)), (255,0,0), 2)



    cv2.line(im_with_keypoints, (0, 380), (500, 380), (255, 0, 0))
    cv2.imshow('fg', im_with_keypoints)
    cv2.imshow('org', frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
