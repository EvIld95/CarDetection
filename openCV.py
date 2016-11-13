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
detector = cv2.SimpleBlobDetector_create(parameters=params)
carKeyPoints = {}
toDelete = []
identifier = 0
distanceAcc = 30
def checkCarInDistanceExists(x,y):
    for carKey in carKeyPoints.keys():
        xx, yy = carKeyPoints[carKey].pt
        if abs(x - xx) < distanceAcc and abs(y - yy) < distanceAcc:
            return True
    return False




while True:
    _, frame = cap.read()
    media = cv2.medianBlur(frame, 3)
    blur = cv2.GaussianBlur(frame, (5, 5), 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask2 = fgbg2.apply(media)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    close = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, kernel)
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel,iterations=2)
    #erode = cv2.erode(fgmask2, (10,10), iterations=3)
    #dilation = cv2.dilate(erode, kernel, iterations=1)
    thresh = 250
    binary = (opening > thresh) * 255
    image = np.uint8(binary)

    keypoints = detector.detect(image)

    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255))
    filtered = list(filter(lambda x: x.pt[1] > 260 and x.pt[1] < 350, keypoints))
    carsInDistance = list(filter(lambda x: x.pt[1] > 260, keypoints))
    for i, keypoint in enumerate(filtered):
        x, y = keypoint.pt

        if checkCarInDistanceExists(x,y) == False:
            carKeyPoints[identifier] = keypoint
            cv2.putText(im_with_keypoints, '{}'.format(identifier), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1,
                            (0, 255, 0), 2)
            identifier = identifier + 1
            # else:
            #     for carK in carKeyPoints.keys():
            #         cx,cy = carKeyPoints[carKey].pt
            #         if abs(cx - x) > 20 or abs(cy - y) > 20:
            #             carKeyPoints[identifier] = keypoint
            #             cv2.putText(im_with_keypoints, '{}'.format(identifier), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            #             identifier = identifier + 1

    for carKey in carsInDistance:
        for key in carKeyPoints.keys():
            newX, newY = carKey.pt
            oldX, oldY = carKeyPoints[key].pt
            if abs(newX - oldX) < distanceAcc//2 and abs(newY - oldY) < distanceAcc:
                if newY > 350:
                    print('USUWAM {}'.format(key))
                    toDelete.append(key)
                else:
                    carKeyPoints[key] = carKey
                    cv2.putText(im_with_keypoints, '{}'.format(key), (int(newX), int(newY)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

    for delete in set(toDelete):
        del(carKeyPoints[delete])

    toDelete = []

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



    cv2.line(im_with_keypoints, (0,350), (600,350), (255,0,0))
    #cv2.imshow('fg',im_with_keypoints)
    cv2.imshow('org',im_with_keypoints)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
