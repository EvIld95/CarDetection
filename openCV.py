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


cap = cv2.VideoCapture('Motorway.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()
fgbg2 = cv2.createBackgroundSubtractorKNN()


params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
params.filterByColor = True

detector = cv2.SimpleBlobDetector_create(parameters=params)

while True:
    _, frame = cap.read()
    #
    media = cv2.medianBlur(frame, 3)
    blur = cv2.GaussianBlur(frame, (9,9),3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask2 = fgbg2.apply(blur)
    sobel = cv2.Canny(fgmask2, 150, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    close = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel)
    #erode = cv2.erode(close, (10,10))
    #erode = cv2.erode(erode, (10, 10))
    #erode = cv2.erode(erode, (10, 10))
    dilation = cv2.dilate(opening, kernel, iterations=2)
    thresh = 100
    binary = (dilation > thresh) * 255
    image = np.uint8(binary)

    #keypoints = detector.detect(image)
    #im_with_keypoints = cv2.drawKeypoints(dilaton, keypoints, np.array([]), (0, 0, 255))

    # treshold = 5
    # for x in range(0, dilaton.shape[0] // treshold):
    #     for y in range(0, dilaton.shape[1] // treshold):
    #         median = calculateMedian(dilaton[treshold * x:(treshold * x) + treshold, treshold * y:(treshold * y) + treshold])
    #
    #
    #         if median > 100:
    #             dilaton[treshold * x:(treshold * x) + treshold, treshold * y:(treshold * y) + treshold] = 255
    #         else:
    #             dilaton[treshold * x:(treshold * x) + treshold, treshold * y:(treshold * y) + treshold] = 0

    cv2.imshow('fg',image)
    cv2.imshow('org',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
