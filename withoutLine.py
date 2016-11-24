from datetime import date, datetime, time

import cv2
import numpy as np


def calculateAveragePixelWhitePower(table):
    sum = 0
    amount = 0
    for x in table:
        for y in x:
            amount += 1
            sum += y
    return sum // amount


cap = cv2.VideoCapture('conv.mov')
width = cap.get(3)
height = cap.get(4)

fgbg2 = cv2.createBackgroundSubtractorKNN()
carCascade = cv2.CascadeClassifier('cars.xml')

params = cv2.SimpleBlobDetector_Params()
params.blobColor = 255
params.filterByColor = True
detector = cv2.SimpleBlobDetector_create(parameters=params)
carKeyPoints = {}
toDelete = []
identifier = 0
distanceAcc = 30
yStartCountingLocation = height // 2
#yDeletObjectLocation = 340 #350
numberOfCars = 0

drawLine = True
frameCounter = 0
def checkCarInDistanceExists(x,y):
    for carKey in carKeyPoints.keys():
        xx, yy, _, _, _  = carKeyPoints[carKey]
        if abs(x - xx) < distanceAcc and abs(y - yy) < distanceAcc:
            return True
    return False


while True:
    _, frame = cap.read()
    media = cv2.medianBlur(frame, 3)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgmask2 = fgbg2.apply(media)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))

    #close = cv2.morphologyEx(fgmask2, cv2.MORPH_CLOSE, kernel, iterations=1)
    #open = cv2.morphologyEx(fgmask2, cv2.MORPH_OPEN, kernel,iterations=1)
    gradient = cv2.morphologyEx(fgmask2, cv2.MORPH_GRADIENT, kernel)

    #erode = cv2.erode(fgmask2, (10,10), iterations=3)
    #dilation = cv2.dilate(erode, kernel, iterations=1)
    thresh = 254
    binary = (gradient > thresh) * 255
    image = np.uint8(binary)

    #keypoints = detector.detect(image)
    #im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255))

    cars = carCascade.detectMultiScale(gray, 1.02, 3)
    realCars = []

    for (x, y, w, h) in cars:
        contains = False
        median = calculateAveragePixelWhitePower(image[y:y + h, x:x + w])
        if median > 50 and (y + w//2) > yStartCountingLocation:
            for sx,sy,_,_ in realCars:
                if (abs(x - sx) < distanceAcc and abs(y - sy) < distanceAcc):
                    contains = True
                    break
            if contains == False:
                realCars.append((x, y, w, h))
                cv2.rectangle(frame, (x, y), (x + w, y + (h)), (255, 0, 0), 2)



    mappedCars = map(lambda tuple: (tuple[0]+(tuple[2]//2), tuple[1]+(tuple[3]//2)), realCars)
    #cv2.putText(frame, 'Samochody ktore przejechaly linie: {}'.format(numberOfCarsMovedLine), (int(20), int(20)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
    cv2.putText(frame, 'Samochody zliczone: {}'.format(numberOfCars), (int(20), int(40)), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)


    carsInDistance = list(filter(lambda x: x[1] > yStartCountingLocation, mappedCars))
    #filtered = list(filter(lambda x: x[1] < yDeletObjectLocation, carsInDistance))
    # for keypoint in mappedCars:
    #     x, y, counter = keypoint
    #     if checkCarInDistanceExists(x,y) == False:
    #         carKeyPoints[identifier] = keypoint
    #         #cv2.putText(frame, '{}'.format(identifier), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1,
    #         #                (0, 255, 0), 2)
    #         identifier = identifier + 1




    for carKey in carsInDistance:
        newX, newY = carKey
        for key in carKeyPoints.keys():
            oldX, oldY, oldCounter, oldPosition, _ = carKeyPoints[key]
            if abs(newX - oldX) < distanceAcc and abs(newY - oldY) < distanceAcc:
                carKeyPoints[key] = (newX, newY, oldCounter, (oldX, oldY), datetime.now())
                cv2.putText(frame, '{}'.format(key), (int(newX), int(newY)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
            # elif abs(oldX - oldPosition[0]) < 1 and abs(oldY - oldPosition[1]) < 1:
            #     if oldCounter > 2:
            #         # numberOfCarsMovedLine = numberOfCarsMovedLine + 1
            #         toDelete.append(key)
            #     carKeyPoints[key] = (oldX,oldY,oldCounter+1, oldPosition)

    for key in carKeyPoints.keys():
        oldX, oldY, oldCounter, oldPosition, lastUpdate = carKeyPoints[key]
        diff = (datetime.now() - lastUpdate).total_seconds()
        if abs(oldX - oldPosition[0]) < 1 and abs(oldY - oldPosition[1]) < 1:
            if oldCounter > 2:
                toDelete.append(key)
            carKeyPoints[key] = (oldX, oldY, oldCounter + 1, oldPosition, lastUpdate)
        elif diff > 2:
            numberOfCars = numberOfCars + 1
            toDelete.append(key)

    for keypoint in carsInDistance:
        x, y = keypoint
        if checkCarInDistanceExists(x, y) == False:
            carKeyPoints[identifier] = (x,y,0, (x,y), datetime.now())
            # cv2.putText(frame, '{}'.format(identifier), (int(x), int(y)), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)
            identifier = identifier + 1

    for delete in set(toDelete):
        print('USUWAM {}'.format(delete))
        del(carKeyPoints[delete])

    toDelete = []

    #cv2.line(frame, (0,yDeletObjectLocation), (int(width),yDeletObjectLocation), (255,0,0))
    cv2.imshow('org', frame)
    frameCounter = frameCounter + 1
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
