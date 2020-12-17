import cv2
import numpy as np
import math
import os
import time
# import matplotlib.pyplot as plt

class DTracker:

    def __init__(self):
        self.frameNumber = 0
        self.objScrThreshold = 0.5 # Probability that the object is contained in the bounding box
        self.classConfThreshold = 0.5 # Probability that the object belongs to a specific class
        self.nmsThreshold = 0.4 # Detect overlapping targets of the same or different types
        self.networkWidth = 416 # Width of network's input image
        self.networkHeight = 416 # Height of network's input image

        self.ball_Left = 0
        self.ball_Top = 0
        self.ball_Right = 0
        self.ball_Bottom = 0

        self.tracker_ball = cv2.TrackerCSRT_create()
        self.tracking_ok_ball = False

        self.tracker_player = cv2.TrackerCSRT_create()
        self.tracking_ok_player = False

        self.classes = None
        self.net = None

        classes_file = "./coco.names"
        model_config = "yolov3.cfg"
        model_weights = "yolov3.weights"

        with open(classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').rsplit('\n')

        self.net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

    def detect(self, img):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # max_wh = max(frame_height, frame_width)
        # max_wh = int(max_wh/32)*32
        # min_wh = min(frame_height, frame_width)
        # min_wh = int(min_wh/32)*32
        # print(max_wh, min_wh)

        # Preprocessing task: 1.Mean subtraction 2.Scaling by some factor
        blob = cv2.dnn.blobFromImage(img, 1/255, (self.networkWidth, self.networkHeight), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Get the name of the YOLO output layer
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Runs the forward pass to get output of the output layers
        layerOutputs = self.net.forward(ln)

        # Init bounding box, confidence and class
        boxes_ball = []
        confidences_ball = []
        classIDs_ball = []

        boxes_player = []
        confidences_player = []
        classIDs_player = []
        centers = []

        for output in layerOutputs:
            for detection in output:
                if detection[4] > self.objScrThreshold:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    # football
                    if self.classes[classID] == "sports ball" and confidence > self.classConfThreshold :
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        width = int(detection[2] * frame_width)
                        height = int(detection[3] * frame_height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)

                        boxes_ball.append([left, top, width, height])
                        confidences_ball.append(float(confidence))
                        classIDs_ball.append(classID)

                    # player
                    if self.classes[classID] == "person" and confidence > self.classConfThreshold:
                        center_x = int(detection[0] * frame_width)
                        center_y = int(detection[1] * frame_height)
                        width = int(detection[2] * frame_width)
                        height = int(detection[3] * frame_height)
                        left = int(center_x - width / 2)
                        top = int(center_y - height / 2)

                        boxes_player.append([left, top, width, height])
                        confidences_player.append(float(confidence))
                        classIDs_player.append(classID)
                        centers.append([center_x, center_y])

        box_ball = None
        box_player = None
        if len(boxes_ball) > 0:
            box_ball = boxes_ball[np.argmax(np.array(confidences_ball))]

            if len(boxes_player) > 0:
                idx = 0
                center_x = box_ball[0] + 2*box_ball[2]
                center_y = box_ball[1] + 2*box_ball[3]
                minDist = frame_width**2 + frame_height**2
                for i in range(len(centers)):
                    dist = (center_x - centers[i][0])**2 + (center_y - centers[i][1])**2
                    if minDist > dist:
                        idx = i
                        minDist = dist
                box_player = boxes_player[idx]

        return box_ball, box_player

    def detect_track(self, img):
        is_detecting_ball = False
        is_detecting_player = False
        if (self.frameNumber % 5 == 0) or (not self.tracking_ok_ball):# or not self.tracking_ok_player:
            ball_box, player_box = self.detect(img)
            # ball_box = self.detect(img)
            if ball_box is not None:
                is_detecting_ball = True
                self.tracker_ball = cv2.TrackerCSRT_create()
                self.tracking_ok_ball = self.tracker_ball.init(img, tuple(ball_box))
                self.tracking_ok_ball = self.tracker_ball.update(img)
            else:
                self.tracking_ok_ball, ball_box = self.tracker_ball.update(img)
            if player_box is not None:
                is_detecting_player = True
                self.tracker_player = cv2.TrackerCSRT_create()
                self.tracking_ok_player = self.tracker_player.init(frame, tuple(player_box))
                self.tracking_ok_player = self.tracker_player.update(frame)
            else:
                self.tracking_ok_player, player_box = self.tracker_player.update(frame)
        else:
            self.tracking_ok_ball, ball_box = self.tracker_ball.update(img)
            self.tracking_ok_player, player_box = self.tracker_player.update(frame)

        self.frameNumber += 1

        ball_box = list(ball_box)
        if ball_box is not None:
            ball_box.append(is_detecting_ball)
        else:
            ball_box = [0, 0, 0, 0, is_detecting_ball]

        player_box = list(player_box)
        if player_box is not None:
            player_box.append(is_detecting_player)
        else:
            player_box = [0, 0, 0, 0, is_detecting_player]

        return ball_box, player_box

testFileName = "test-s.mp4"
resultFileName = "result.mp4"

if __name__ == "__main__":
    DT = DTracker()

    cap = cv2.VideoCapture(testFileName)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    print("FPS:",fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(resultFileName, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    flag, frame = cap.read()
    while flag:
        ball_box, player_box = DT.detect_track(frame)
        # ball_box = DT.detect_track(frame)
        result = frame

        left = int(ball_box[0])
        top = int(ball_box[1])
        width = int(ball_box[2])
        height = int(ball_box[3])
        if ball_box[4]:
            result = cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 3)
        else:
            result = cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 178, 50), 3)

        left = int(player_box[0])
        top = int(player_box[1])
        width = int(player_box[2])
        height = int(player_box[3])
        if player_box[4]:
            result = cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 3)
        else:
            result = cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 178, 50), 3)
        out.write(np.uint8(result))
        cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        flag, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()