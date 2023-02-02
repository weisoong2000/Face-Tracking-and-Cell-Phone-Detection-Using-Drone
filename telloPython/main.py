import logging
import time
import cv2
import cvzone
import numpy as np
from djitellopy import tello
from time import sleep

thres = 0.60
nmsThres = 0.2
classNames = []
classFile = 'ss.names' # Contains a totoal of 91 different objects which can be recognized by the code
with open(classFile, 'rt') as f:
    classNames = f.read().split('\n')

print(classNames)
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = "frozen_inference_graph.pb"
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def findFace(img):
    faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")
    faceCascade.load("Resources/haarcascade_frontalface_default.xml")
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 8)
    myFaceListC = []
    myFaceListArea = []

    for (x, y, w, h) in faces:
        cv2.rectangle(img,(x,y),(x+w, y+h),(0, 0, 255),2)
        cx = x + w//2
        cy = y + h//2
        area = w*h
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        myFaceListC.append([cx,cy])
        myFaceListArea.append(area)
    if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        InfoText = "Area:{0} X:{1} Y:{2}".format(area , cx, cy)
        cv2.putText(img, InfoText, (cx+20, cy), font, fontScale, fontColor, lineThickness)
        return img, [myFaceListC[i], myFaceListArea[i]]
    else:
        return img, [[0, 0], 0]

def trackFace(me, img, info, w, h, pid, DetectRange, pErrorRotate, pErrorUp):
    area = info[1]
    x, y = info[0][0], info[0][1]
    fb = 0
    ErrorRotate = x - w/2
    ErrorUp = h/2 - y
    cv2.circle(img, (int(w/2), int(h/2)), 5, (0, 255, 0), cv2.FILLED)
    if x >10 or y >10:
        cv2.line(img, (int(w/2), int(h/2)), (x,y), (255, 0, 0), lineThickness)
    rotatespeed = pid[0]*ErrorRotate + pid[1]*(ErrorRotate - pErrorRotate)
    updownspeed = pid[0]*ErrorUp + pid[1]*(ErrorUp - pErrorUp)
    rotatespeed = int(np.clip(rotatespeed, -40, 40))
    updownspeed = int(np.clip(updownspeed, -60, 60))

    area = info[1]
    if area > DetectRange[0] and area < DetectRange[1]:
        fb = 0
        # updownspeed = 0
        # rotatespeed = 0
        InfoText = "Hold Speed:{0} Rotate:{1} Up:{2}".format(fb, rotatespeed, updownspeed)
        cv2.putText(img, InfoText, (10, 60), font, fontScale, fontColor, lineThickness)
        me.send_rc_control(0, fb, updownspeed, rotatespeed)
    elif area > DetectRange[1]:
        fb = -20
        InfoText = "Backward Speed:{0} Rotate:{1} Up:{2}".format(fb, rotatespeed, updownspeed)
        cv2.putText(img, InfoText, (10, 60), font, fontScale, fontColor, lineThickness)
        me.send_rc_control(0, fb, updownspeed, rotatespeed)
    elif area < DetectRange[0] and area > 1000:
        fb = 20
        InfoText = "Forward Speed:{0} Rotate:{1} Up:{2}".format(fb, rotatespeed, updownspeed)
        cv2.putText(img, InfoText, (10, 60), font, fontScale, fontColor, lineThickness)
        me.send_rc_control(0, fb, updownspeed, rotatespeed)
    else:
        me.send_rc_control(0, 0, 0, 0)

    if x == 0:
        speed = 0
        error = 0
    return ErrorRotate, ErrorUp


# Main Program

# Camera Setting
Camera_Width = 720
Camera_Height = 480
DetectRange = [6000, 11000]  # DetectRange[0] 是保持静止的检测人脸面积阈值下限，DetectRange[0] 是保持静止的检测人脸面积阈值上限
PID_Parameter = [0.5, 0.0004, 0.4]
pErrorRotate, pErrorUp = 0, 0

# Font Settings
font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
fontColor = (255, 0, 0)
lineThickness = 1

# Tello Init
Drone = tello.Tello()  # 创建飞行器对象
Drone.connect()  # 连接到飞行器
Drone.streamon()  # 开启视频传输
#Drone.LOGGER.setLevel(logging.ERROR)  # 只显示错误信息
sleep(5)  #  等待视频初始化
Drone.takeoff()
sleep(2)
Drone.move_up(70)
sleep(2)

con = True
con1 = True

while con:
    OriginalImage = Drone.get_frame_read().frame
    Image = cv2.resize(OriginalImage, (Camera_Width, Camera_Height))
    img, info = findFace(Image)
    pErrorRotate, pErrorUp = trackFace(Drone, img, info, Camera_Width, Camera_Height, PID_Parameter, DetectRange, pErrorRotate, pErrorUp)
    sleep(1)
    # cv2.imshow("Drone Control Centre", Image)
    # cv2.waitKey(1)
    classIds, confs, bbox = net.detect(OriginalImage, confThreshold=thres, nmsThreshold=nmsThres)  # To remove duplicates / declare accuracy

    while con1:

        try:
                for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    # sleep(1)
                    # Drone.rotate_clockwise(45)

                    if classNames[classId - 1] == 'person':
                        cvzone.cornerRect(OriginalImage, box)
                        cv2.putText(OriginalImage, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                                    (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                    1, (0, 255, 0), 2)
                        con1 = False
                        print('person detected!')
                        sleep(2)
                        # cv2.imwrite("person.png", OriginalImage)  # Taking picture
                        break
                    else:
                        # Drone.rotate_clockwise(45)
                        # Drone.curve_xyz_speed(500,500,500,500,500,500,10)
                        sleep(5)

        except:
                pass


    try:
            for classId, conf, box in zip(classIds.flatten(), confs.flatten(), bbox):

                if classNames[classId - 1] == 'cell phone':
                    cvzone.cornerRect(OriginalImage, box)
                    cv2.putText(OriginalImage, f'{classNames[classId - 1].upper()} {round(conf * 100, 2)}',
                                (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                1, (0, 255, 0), 2)
                    sleep(2)
                    Drone.flip_back()
                    cv2.imwrite("picture.png", OriginalImage)  # Taking picture
                    sleep(1)
                    Drone.land()
                    con=False
                    break


    except:
        pass

    cv2.imshow("Drone Control Centre", Image)
    cv2.waitKey(1)

