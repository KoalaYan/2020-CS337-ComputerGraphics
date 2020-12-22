import cv2
import numpy as np
import math
import os
import time
import judgeIn
import persp
import multiTracker
import edgeCorner
import teamClassify
import pandas as pd
# import matplotlib.pyplot as plt

class DTracker:

    def __init__(self):
        self.frameNumber = 0
        self.objScrThreshold = 0.5 # Probability that the object is contained in the bounding box
        self.classConfThreshold = 0.5 # Probability that the object belongs to a specific class
        self.nmsThreshold = 0.35 # Detect overlapping targets of the same or different types
        self.networkWidth = 416 # Width of network's input image
        self.networkHeight = 416 # Height of network's input image
        self.vertex_lst = []

        self.classes = None
        self.net = None

        classes_file = "./coco.names"
        model_config = "yolov4.cfg"
        model_weights = "yolov4.weights"

        with open(classes_file, 'rt') as f:
            self.classes = f.read().rstrip('\n').rsplit('\n')

        self.net = cv2.dnn.readNetFromDarknet(model_config, model_weights)

        # 'BOOSTING', 'MIL', 'KCF','TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT'
        self.trackerType = "CSRT"
        self.multiTracker = cv2.MultiTracker_create()
        self.isTracking = False

    # Non-maximum suppression (nms) algorithm
    def nms(self, dets, scores_list):
        thresh = self.nmsThreshold
        dets = np.array(dets, np.float)
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = x1 + dets[:, 2]
        y2 = y1 + dets[:, 3]
        scores = np.array(scores_list, np.float)

        areas = (x2 - x1 + 1) * (y2 - y1 + 1) #all boxes' area
        order = scores.argsort()[::-1] # index in score descending

        keep = []
        while order.size > 0:
            i = order[0] # argmax score index box coordinate
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]]) # most score box intersect other boxes

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1) # height width
            inter = w * h # total ares of other all boxes
            ovr = inter / (areas[i] + areas[order[1:]] - inter)  # IOU: intersection/union

            inds = np.where(ovr <= thresh)[0] # lower ovr, smaller intersection. Maybe another one
            order = order[inds + 1]  # iou < threshold

        return keep


    def detect(self, frame):
        self.frameNumber += 1
        # print(self.frameNumber)
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        max_wh = max(frame_height, frame_width)
        max_wh = int(max_wh/32)*32
        min_wh = min(frame_height, frame_width)
        min_wh = int(min_wh/32)*32
        # print(max_wh, min_wh)

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
        # classIDs_player = []
        # centers = []

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
                        # classIDs_player.append(classID)
                        # centers.append([center_x, center_y])

        idx = self.nms(boxes_player, confidences_player)
        res_player = []
        for i in idx:
            # res_player.append(boxes_player[i])

            box = boxes_player[i]
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])
            poi = [left, top+height]
            if judgeIn.isin_multipolygon(poi, self.vertex_lst, contain_boundary=True):
                # result = cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 3)
                res_player.append(box)

        return res_player


    def detect_tracker(self, frame):
        is_detecting = False
        if(self.frameNumber % 5 == 0) or (not self.isTracking):
            boxes_list = self.detect(frame)
            if len(boxes_list) != 0:
                is_detecting = True
                self.multiTracker = cv2.MultiTracker_create()
                for box in boxes_list:
                    self.multiTracker.add(multiTracker.createTrackerByName(self.trackerType), frame, (box[0],box[1],box[2],box[3]))
                self.isTracking = True
            else:
                self.isTracking, boxes_list = self.multiTracker.update(frame)
                boxes_list = boxes_list.tolist()
        else:
            self.isTracking, boxes_list = self.multiTracker.update(frame)
            boxes_list = boxes_list.tolist()
            self.frameNumber += 1
            # print(self.frameNumber)
        # print(boxes_list)
        return boxes_list


def dataLog(data_list, team_label, filename1, filename2):
    if len(data_list) != len(team_label):
        print("Error data log!")
        return
    # indata = data_list
    fp1=open(filename1,"a+",encoding="utf-8")
    fp2=open(filename2,"a+",encoding="utf-8")
    for idx in range(0, len(data_list)):
        data = data_list[idx]
        if team_label[idx] == 0:
            fp1.write(str(data[0])+','+str(data[1]))
            # if idx != len(data_list) - 1:
            fp1.write('|')
        elif team_label[idx] == 1:
            fp2.write(str(data[0])+','+str(data[1]))
            # if idx != len(data_list) - 1:
            fp2.write('|')
        else:
            print("error log!")

    fp1.write('\n')
    fp1.close()
    fp2.write('\n')
    fp2.close()

testFileName = "test.mp4"
resultFileName = "color.mp4"
logFileName1 = "data_1.log"
logFileName2 = "data_2.log"

if __name__ == "__main__":

    DT = DTracker()

    cap = cv2.VideoCapture(testFileName)
    fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    print("FPS:",fps)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    # Define the codec and create VideoWriter object.
    out = cv2.VideoWriter(resultFileName, cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width, frame_height))

    flag, img = cap.read()
    if flag:
        lt, rt, rd, ld = edgeCorner.corner_detect(img)
        # lt[0] = 0
        # lt[1] = lt[1] - 300
        # rt[0] = img.shape[1] - 1
        # rt[1] = rt[1] - 300
        # rd[0] = img.shape[1] - 1
        # rd[1] = rd[1] - 300
        # ld[0] = 0
        # ld[1] = ld[1] - 300
        lt[0] = lt[0] - 300
        lt[1] = lt[1] - 300
        rt[0] = rt[0] - 300
        rt[1] = rt[1] - 300
        rd[0] = rd[0] - 300
        rd[1] = rd[1] - 300
        ld[0] = ld[0] - 300
        ld[1] = ld[1] - 300
        # print(lt, rt, rd, ld)
        DT.vertex_lst = [ld, rd, rt, lt]
    team_list = []

    while flag:
        player_boxes = DT.detect_tracker(img)
        print("Current Frame:", DT.frameNumber)
        # if (DT.frameNumber % 5 == 0) or (len(team_list) == 0):
        team_list = teamClassify.teamClassify_kmeans(img, player_boxes)
        # print("Number:", len(player_boxes))
        result = img

        point_list = []

        for idx in range(len(player_boxes)):
            box = player_boxes[idx]
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])
            poi = [left, top+height]
            point_list.append(poi)

        result, point_list = persp.persp(img, point_list)
        # print(len(point_list))

        # data_list = []
        # for idx in range(0, len(point_list)):
        #     poi = point_list[idx]
        #     poi.append(team_list[idx][4])
        #     data_list.append(poi)

        dataLog(point_list, np.array(team_list,np.int)[:,4], logFileName1, logFileName2)

        # print(result.shape)
        for idx in range(0,len(point_list)):
            # print(poi)
            poi = point_list[idx]
            team = team_list[idx][4]
            if team == 1:
                result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (0, 0, 255), 3)
            elif team == 0:
                result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (255, 0, 0), 3)
            else:
                result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (255, 0, 0), 3)

        print("Player Number:",len(player_boxes))
        #
        # point_size = 1
        # point_color = (0, 0, 255) # BGR
        # thickness = 4 # 0/4/8
        #
        # for poi in point_list:
        #     result = cv2.circle(result, (poi[0],poi[1]), point_size, point_color, thickness)

        out.write(np.uint8(result))
        cv2.imshow('result', result)
        # cv2.imwrite('res-test.jpg', result)
        # if DT.frameNumber > 10:
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        flag, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()