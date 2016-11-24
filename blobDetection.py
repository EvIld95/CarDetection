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

fgbg2 = cv2.createBackgroundSubtractorKNN()
carCascade = cv2.CascadeClassifier('Cascade.xml')

params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
params.filterByColor = True
detector = cv2.SimpleBlobDetector_create(parameters=params)
carKeyPoints = {}
toDelete = []
identifier = 0
distanceAcc = 30
numberOfCarsMovedLine = 0
def checkCarInDistanceExists(x,y):
    for carKey in carKeyPoints.keys():
        xx, yy = carKeyPoints[carKey].pt
        if abs(x - xx) < distanceAcc and abs(y - yy) < distanceAcc:
            return True
    return False


startCounting = True

while True:
    _, frame = cap.read()
    media = cv2.medianBlur(frame, 3)
    blur = cv2.GaussianBlur(frame, (5, 5), 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask2 = fgbg2.apply(media)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))

    close = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, kernel, iterations=1)
    opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel,iterations=2)
    #erode = cv2.erode(fgmask2, (10,10), iterations=3)
    #dilation = cv2.dilate(erode, kernel, iterations=1)
    thresh = 254
    binary = (opening > thresh) * 255
    image = np.uint8(binary)

    keypoints = detector.detect(image)

    im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0, 0, 255))

    if startCounting == True:
        filtered = list(filter(lambda x: x.pt[1] > 240 and x.pt[1] < 350, keypoints))
        carsInDistance = list(filter(lambda x: x.pt[1] > 240, keypoints))
        cv2.putText(im_with_keypoints, 'Samochody ktore przejechaly linie: {}'.format(numberOfCarsMovedLine), (int(20), int(20)),cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        cv2.putText(im_with_keypoints, 'Samochody zaindeksowane: {}'.format(identifier), (int(20), int(40)), cv2.FONT_HERSHEY_PLAIN,1, (255, 0, 0), 1)

        for i, keypoint in enumerate(filtered):
            x, y = keypoint.pt

            if checkCarInDistanceExists(x,y) == False:
                carKeyPoints[identifier] = keypoint
                cv2.putText(im_with_keypoints, '{}'.format(identifier), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1,
                                (0, 255, 0), 2)
                identifier = identifier + 1


        for carKey in carsInDistance:
            for key in carKeyPoints.keys():
                newX, newY = carKey.pt
                oldX, oldY = carKeyPoints[key].pt
                if abs(newX - oldX) < distanceAcc//2 and abs(newY - oldY) < distanceAcc:
                    if newY > 350:
                        print('USUWAM {}'.format(key))
                        numberOfCarsMovedLine = numberOfCarsMovedLine + 1
                        toDelete.append(key)
                    else:
                        carKeyPoints[key] = carKey
                        cv2.putText(im_with_keypoints, '{}'.format(key), (int(newX), int(newY)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)

        for delete in set(toDelete):
            del(carKeyPoints[delete])

        toDelete = []


    cv2.line(im_with_keypoints, (0,350), (600,350), (255,0,0))
    cv2.imshow('fg',im_with_keypoints)
    #cv2.imshow('org',image)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()