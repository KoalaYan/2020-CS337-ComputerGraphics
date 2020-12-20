import cv2
import numpy as np
from team_test import input_for_classify_team


def teamClassify():
    image, box_list = input_for_classify_team()
    # 图片转化成RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(box_list)
    info_list = []
    red_count = 0
    blue_count = 0
    rest_count = 0

    for box in box_list:
        # 位置信息
        left = int(box[0]); top = int(box[1]); width = int(box[2]); height = int(box[3])
        # 求颜色均值
        meanColor = [0, 0, 0]
        count = 0
        for i in range(left, left + width + 1, 1):
            for j in range(top, top + height + 1, 1):
                pixel = image[j][i]
                R = int(pixel[0]); G = int(pixel[1]); B = int(pixel[2])

                if G - R > 5 and G - B > 5:
                    # 为绿色，跳过该像素点
                    continue
                # 计入该点
                meanColor[0] += R; meanColor[1] += G; meanColor[2] += B
                count += 1
        if count == 0:
            print("Empty Box!")
        else:
            meanColor[0] /= count
            meanColor[1] /= count
            meanColor[2] /= count
        print(meanColor)
        R = int(meanColor[0]); G = int(meanColor[1]); B = int(meanColor[2])
        if R - G > 5 and R - B > 5:
            red_count += 1
        elif B - R > 0 and B - G > 0:
            blue_count += 1
        else:
            rest_count += 1
        draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), (R, G, B), 2)

    print("Red:", red_count, "Blue:", blue_count, "Rest:", rest_count)
    cv2.imwrite("result.jpg", image)

teamClassify()
