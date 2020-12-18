import cv2
import numpy as np
import math
import os
import time
import judgeIn
# import matplotlib.pyplot as plt

class DTracker:

    def __init__(self):
        self.frameNumber = 0
        self.objScrThreshold = 0.5 # Probability that the object is contained in the bounding box
        self.classConfThreshold = 0.5 # Probability that the object belongs to a specific class
        self.nmsThreshold = 0.4 # Detect overlapping targets of the same or different types
        self.networkWidth = 416 # Width of network's input image
        self.networkHeight = 416 # Height of network's input image

        self.classes = None
        self.net = None

        classes_file = "./coco.names"
        model_config = "yolov4.cfg"
        model_weights = "yolov4.weights"

        with open(classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').rsplit('\n')

        self.net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

    def detect(self, frame):
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        max_wh = max(frame_height, frame_width)
        max_wh = int(max_wh/32)*32
        min_wh = min(frame_height, frame_width)
        min_wh = int(min_wh/32)*32
        print(max_wh, min_wh)

        # Preprocessing task: 1.Mean subtraction 2.Scaling by some factor
        #blob = cv2.dnn.blobFromImage(frame, 1/255, (self.networkWidth, self.networkHeight), [0,0,0], 1, crop=False)
        blob = cv2.dnn.blobFromImage(frame, 1/255, (max_wh, min_wh), [0,0,0], 1, crop=False)

        # Sets the input to the network
        self.net.setInput(blob)

        # Get the name of the YOLO output layer
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Runs the forward pass to get output of the output layers
        layerOutputs = self.net.forward(ln)

        # Init bounding box, confidence and class
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

        return boxes_player

    # def detect_track(self, frame):
    #     is_detecting_ball = False
    #     is_detecting_player = False
    #     if (self.frameNumber % 5 == 0) or (not self.tracking_ok_ball):# or not self.tracking_ok_player:
    #         ball_box, player_box = self.detect(frame)
    #         # ball_box = self.detect(img)
    #         if ball_box is not None:
    #             is_detecting_ball = True
    #             self.tracker_ball = cv2.TrackerCSRT_create()
    #             self.tracking_ok_ball = self.tracker_ball.init(frame, tuple(ball_box))
    #             self.tracking_ok_ball = self.tracker_ball.update(frame)
    #         else:
    #             self.tracking_ok_ball, ball_box = self.tracker_ball.update(frame)
    #         if player_box is not None:
    #             is_detecting_player = True
    #             self.tracker_player = cv2.TrackerCSRT_create()
    #             self.tracking_ok_player = self.tracker_player.init(frame, tuple(player_box))
    #             self.tracking_ok_player = self.tracker_player.update(frame)
    #         else:
    #             self.tracking_ok_player, player_box = self.tracker_player.update(frame)
    #     else:
    #         self.tracking_ok_ball, ball_box = self.tracker_ball.update(frame)
    #         self.tracking_ok_player, player_box = self.tracker_player.update(frame)
    #
    #     self.frameNumber += 1
    #
    #     ball_box = list(ball_box)
    #     if ball_box is not None:
    #         ball_box.append(is_detecting_ball)
    #     else:
    #         ball_box = [0, 0, 0, 0, is_detecting_ball]
    #
    #     player_box = list(player_box)
    #     if player_box is not None:
    #         player_box.append(is_detecting_player)
    #     else:
    #         player_box = [0, 0, 0, 0, is_detecting_player]
    #
    #     return ball_box, player_box

    def persp(self, img):
        lt = [323,398]
        rt = [1593,408]
        ld = [0,675]
        rd = [1910,693]
        h = img.shape[0]
        w = img.shape[1]
        print(w, h)
        point1 = np.array([lt,rt,rd,ld],dtype = "float32")
        point2 = np.array([[0,0],[w-1,0],[w,h],[0,h]],dtype = "float32")
        M = cv2.getPerspectiveTransform(point1,point2)
        out_img = cv2.warpPerspective(img,M,(w,h))

        return out_img
testFileName = "1.mp4"
resultFileName = "field.mp4"

if __name__ == "__main__":
    lt = [323,398]
    rt = [1593,408]
    ld = [0,675]
    rd = [1910,693]
    vertex_lst = [ld, rd, rt, lt]
    DT = DTracker()

    cap = cv2.VideoCapture(testFileName)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    print("FPS:",fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(resultFileName, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    flag, img = cap.read()
    while flag:
        player_boxes = DT.detect(img)
        result = img

        for box in player_boxes:
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])
            poi = [left, top+height]
            if judgeIn.isin_multipolygon(poi,vertex_lst, contain_boundary=True):
                result = cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 3)
            # if box[4]:
            #     result = cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 3)
            # else:
            #     result = cv2.rectangle(img, (left, top), (left + width, top + height), (255, 178, 50), 3)

        # result = DT.persp(result)
        out.write(np.uint8(result))
        cv2.imshow('result', result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        flag, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()