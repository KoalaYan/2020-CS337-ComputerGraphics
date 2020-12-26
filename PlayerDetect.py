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

        self.confront = 0 # Number of players confrontation
        self.confrtThreshold = 5

        self.classes = None
        self.net = None

        self.team_color_1 = (0,0,255)
        self.team_color_2 = (255,0,0)

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
        if(self.frameNumber % 125 == 0) or (not self.isTracking):
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


def dataLog(rec_team1, rec_team2, filename1, filename2):
    # indata = data_list
    fp1=open(filename1,"a+",encoding="utf-8")
    fp2=open(filename2,"a+",encoding="utf-8")
    for poi in rec_team1:
        fp1.write(str(poi[0])+','+str(poi[1]))
        fp1.write('|')

    for poi in rec_team2:
        fp2.write(str(poi[0])+','+str(poi[1]))
        fp2.write('|')

    fp1.write('\n')
    fp1.close()
    fp2.write('\n')
    fp2.close()


testFileName = "1.mp4"
resultFileName = "pre.mp4"
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

    rec_team1 = []
    rec_team2 = []
    pre_cent1 = [0,0]
    pre_cent2 = [0,0]
    team_list = []

    pre_idx_1 = []
    pre_idx_2 = []

    while flag:
        player_boxes = DT.detect_tracker(img)
        print("Current Frame:", DT.frameNumber)
        # if (DT.frameNumber % 5 == 0) or (len(team_list) == 0):
        team_list = teamClassify.teamClassify_kmeans(img, player_boxes)
        # print("Number:", len(player_boxes))
        result = img.copy()

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



        rec_team1 = []
        rec_team2 = []
        show_team1 = []
        show_team2 = []

        if DT.frameNumber != 1:
            if DT.frameNumber % 125 != 1:
                if len(point_list) != len(pre_idx_1) + len(pre_idx_2):
                    flag, img = cap.read()
                    DT.frameNumber += 124 - ((DT.frameNumber-1) % 125)
                    continue
                for idx in pre_idx_1:
                    rec_team1.append(point_list[idx])
                    show_team1.append(player_boxes[idx])
                for idx in pre_idx_2:
                    rec_team2.append((point_list[idx]))
                    show_team2.append(player_boxes[idx])

                centroid1 = np.mean(rec_team1, axis=0)
                centroid2 = np.mean(rec_team2, axis=0)
                pre_cent1 = centroid1
                pre_cent2 = centroid2
                # print("location:", centroid1, centroid2)

            else:
                pre_idx_1 = []
                pre_idx_2 = []
                label_1 = team_list[0][4]
                for clr in np.array(team_list)[:,4]:
                    if any(clr != label_1):
                        label_2 = clr
                        break
                for idx in range(len(point_list)):
                    if all(team_list[idx][4] == label_1):
                        rec_team1.append(point_list[idx])
                        show_team1.append(player_boxes[idx])
                        pre_idx_1.append(idx)
                    else:
                        rec_team2.append(point_list[idx])
                        show_team2.append(player_boxes[idx])
                        pre_idx_2.append(idx)
                centroid1 = np.mean(rec_team1, axis=0)
                centroid2 = np.mean(rec_team2, axis=0)
                dist1 = np.sqrt(np.sum(np.square(centroid1-pre_cent1))) + np.sqrt(np.sum(np.square(centroid2-pre_cent2)))
                dist2 = np.sqrt(np.sum(np.square(centroid1-pre_cent2))) + np.sqrt(np.sum(np.square(centroid2-pre_cent1)))
                # print("location:", centroid1, centroid2)
                # print("distance:", dist1, dist2)
                if dist1 > dist2:
                    rec_team1, rec_team2 = rec_team2, rec_team1
                    show_team1, show_team2 = show_team2, show_team1
                    pre_idx_1, pre_idx_2 = pre_idx_2, pre_idx_1
                    pre_cent1 = centroid2
                    pre_cent2 = centroid1
                else:
                    pre_cent1 = centroid1
                    pre_cent2 = centroid2

                print(pre_idx_1, pre_idx_2)
        else:
            pre_idx_1 = []
            pre_idx_2 = []

            label_1 = team_list[0][4]
            # print(label_1)
            DT.team_color_1 = label_1
            # DT.team_color_1 = (label_1[0],label_1[1],label_1[2])
            for clr in np.array(team_list)[:,4]:
                if any(clr != label_1):
                    label_2 = clr
                    # print(label_2)
                    DT.team_color_2 = label_2
                    # DT.team_color_2 = (label_2[0],label_2[1],label_2[2])
                    break
            for idx in range(len(point_list)):
                if all(team_list[idx][4] == label_1):
                    rec_team1.append(point_list[idx])
                    show_team1.append(player_boxes[idx])
                    pre_idx_1.append(idx)
                else:
                    rec_team2.append(point_list[idx])
                    show_team2.append(player_boxes[idx])
                    pre_idx_2.append(idx)

            print(pre_idx_1, pre_idx_2)

            pre_cent1 = np.mean(rec_team1, axis=0)
            pre_cent2 = np.mean(rec_team2, axis=0)

        # centroid1 = np.mean(rec_team1, axis=0)
        # centroid2 = np.mean(rec_team2, axis=0)
        # if centroid1[0] > centroid2[0]:
        #     rec_team1, rec_team2 = rec_team2, rec_team1

        dataLog(rec_team1, rec_team2, logFileName1, logFileName2)

        # print(DT.team_color_1, DT.team_color_2)

        # Confront.
        # if DT.frameNumber % 25 == 0:
        #     for pl_1 in rec_team1:
        #         for pl_2 in rec_team2:
        #             dist = np.sqrt(np.sum(np.square(pl_1-pl_2)))
        #             if dist < DT.confrtThreshold:
        #                 DT.confront += 1

        for box in show_team1:
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])
            img = cv2.rectangle(img, (left, top), (left + width, top + height), (int(DT.team_color_1[2]),int(DT.team_color_1[1]),int(DT.team_color_1[0])), 2)
        for box in show_team2:
            left = int(box[0])
            top = int(box[1])
            width = int(box[2])
            height = int(box[3])
            img = cv2.rectangle(img, (left, top), (left + width, top + height), (int(DT.team_color_2[2]),int(DT.team_color_2[1]),int(DT.team_color_2[0])), 2)



        # for poi in rec_team1:
        #     result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (int(DT.team_color_1[2]),int(DT.team_color_1[1]),int(DT.team_color_1[0])), 3)
        # for poi in rec_team2:
        #     result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (int(DT.team_color_2[2]),int(DT.team_color_2[1]),int(DT.team_color_2[0])), 3)

        # for idx in range(0,len(point_list)):
        #     # print(poi)
        #     poi = point_list[idx]
        #     team = team_list[idx][4]
        #     if team == 1:
        #         result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (0, 0, 255), 3)
        #     elif team == 0:
        #         result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (255, 0, 0), 3)
        #     else:
        #         result = cv2.rectangle(result, (poi[0], poi[1]), (poi[0]+5, poi[1]+5), (255, 0, 0), 3)

        print("Player Number:",len(player_boxes))
        #
        # point_size = 1
        # point_color = (0, 0, 255) # BGR
        # thickness = 4 # 0/4/8
        #
        # for poi in point_list:
        #     result = cv2.circle(result, (poi[0],poi[1]), point_size, point_color, thickness)

        out.write(np.uint8(img))
        cv2.imshow('result', img)
        # cv2.imwrite('res-test.jpg', result)
        # if DT.frameNumber > 10:
        #     break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        flag, img = cap.read()

    cap.release()
    cv2.destroyAllWindows()