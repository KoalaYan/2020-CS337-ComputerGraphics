from random import random

import cv2
import numpy as np
from team_test import input_for_classify_team
from math import sqrt

color_list = []

image, box_list = input_for_classify_team()
# 图片转化成RGB格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def detect_color(left, top, width, height, image):
    meanColor = [0, 0, 0]
    count = 0
    for i in range(left, left + width + 1, 1):
        for j in range(top, top + height + 1, 1):
            pixel = image[j][i]
            R = int(pixel[0]); G = int(pixel[1]); B = int(pixel[2])
            if (G - R > 5 and G - B > 5):
                # 为绿色，跳过该像素点
                continue
            # 计入该点
            meanColor[0] += R; meanColor[1] += G; meanColor[2] += B
            count += 1
    if count == 0:
        print("Empty Box!!")
    else:
        meanColor[0] /= count
        meanColor[1] /= count
        meanColor[2] /= count
    R = int(meanColor[0]); G = int(meanColor[1]); B = int(meanColor[2])
    if R - G > 5 and R - B > 5:
        return (255, 0, 0)
    elif B - R > 0 and B - G > 0:
        return (0, 0, 255)
    else:
        return (0, 255, 255)

def teamClassify(image, box_list):
    #image, box_list = input_for_classify_team()
    # 图片转化成RGB格式
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(box_list)
    #info_list = []
    red_count = 0
    blue_count = 0
    rest_count = 0

    for i in range(0, len(box_list), 1):
        box = box_list[i]
        # 位置信息
        left = int(box[0]); top = int(box[1]); width = int(box[2]); height = int(box[3])
        # 求颜色均值
        meanColor = [0, 0, 0]
        count = 0
        for i in range(left, left + width + 1, 1):
            for j in range(top, top + height + 1, 1):
                pixel = image[j][i]
                R = int(pixel[0]); G = int(pixel[1]); B = int(pixel[2])

                if (G - R > 5 and G - B > 5):
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
            # 对RGB三维向量进行伸展操作，以提高聚类精度，但是可能会对人工阈值产生影响
            meanColor[0] = (meanColor[0] - 130) * 100000
            meanColor[1] = (meanColor[1] - 130) * 100000
            meanColor[2] = (meanColor[2] - 130) * 100000
        color_list.append(meanColor)

        R = int(meanColor[0]); G = int(meanColor[1]); B = int(meanColor[2])
        if R - G > 5 and R - B > 5:
            red_count += 1
        elif B - R > 0 and B - G > 0:
            blue_count += 1
        else:
            rest_count += 1
        #draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), (R, G, B), 2)

    print("Red:", red_count, "Blue:", blue_count, "Rest:", rest_count)
    #cv2.imwrite("result.jpg", image)

teamClassify(image, box_list)

# 计算一个样本与数据集中所有样本的欧氏距离的平方
def euclidean_distance(one_sample, X):
    one_sample = one_sample.reshape(1, -1)
    X = X.reshape(X.shape[0], -1)
    distances = np.power(np.tile(one_sample, (X.shape[0], 1)) - X, 2).sum(axis=1)
    return distances

class Kmeans():
    """Kmeans聚类算法.
    Parameters:
    -----------
    k: int
        聚类的数目.
    max_iterations: int
        最大迭代次数.
    varepsilon: float
        判断是否收敛, 如果上一次的所有k个聚类中心与本次的所有k个聚类中心的差都小于varepsilon,
        则说明算法已经收敛
    """

    def __init__(self, k=2, max_iterations=500, varepsilon=0.0001):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[i] = centroid
        return centroids

    # 返回距离该样本最近的一个中心索引[0, self.k)
    def _closest_centroid(self, sample, centroids):
        distances = euclidean_distance(sample, centroids)
        closest_i = np.argmin(distances)
        return closest_i

    # 将所有样本进行归类，归类规则就是将该样本归类到与其最近的中心
    def create_clusters(self, centroids, X):
        n_samples = np.shape(X)[0]
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    # 对中心进行更新
    def update_centroids(self, clusters, X):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroid = np.mean(X[cluster], axis=0)
            centroids[i] = centroid
        return centroids

    # 将所有样本进行归类，其所在的类别的索引就是其类别标签
    def get_cluster_labels(self, clusters, X):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        # 从所有样本中随机选取self.k样本作为初始的聚类中心
        centroids = self.init_random_centroids(X)

        # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
        for _ in range(self.max_iterations):
            # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
            clusters = self.create_clusters(centroids, X)
            former_centroids = centroids

            # 计算新的聚类中心
            centroids = self.update_centroids(clusters, X)

            # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
            diff = centroids - former_centroids
            if diff.any() < self.varepsilon:
                break

        return self.get_cluster_labels(clusters, X)

Clf = Kmeans(k=3)
color_list = np.mat(color_list)
y_pred = Clf.predict(color_list)
print(y_pred)

# 每个聚类里的颜色是否已经被探知
zero_is_detected = False
one_is_detected = False
two_is_detected = False
# 颜色
red = (255, 0, 0)
blue = (0, 0, 255)
yellow = (0, 255, 255)
# 每个聚类的颜色（默认都为红）
zeroColor = red
oneColor = red
twoColor = red

for i in range(0, len(y_pred), 1):
    box = box_list[i]
    left = int(box[0]); top = int(box[1]); width = int(box[2]); height = int(box[3])

    # 球员阵营绘制
    if y_pred[i] == 0:
        if not zero_is_detected:
            zeroColor = detect_color(left, top, width, height, image)
            zero_is_detected = True
        draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), zeroColor, 2)
        #draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), red, 2)

    if y_pred[i] == 1:
        if not one_is_detected:
            oneColor = detect_color(left, top, width, height, image)
            one_is_detected = True
        draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), oneColor, 2)
        #draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), blue, 2)

    if y_pred[i] == 2:
        if not two_is_detected:
            twoColor = detect_color(left, top, width, height, image)
            two_is_detected = True
        draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), twoColor, 2)
        #draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), yellow, 2)


cv2.imwrite("result.jpg", image)