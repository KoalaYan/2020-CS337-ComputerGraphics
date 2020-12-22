import cv2
import numpy as np
import edgeCorner
import random


def img_masked_rev(image):
    #converting into hsv image
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_green = np.array([40,40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    mask = ~ mask
    res = cv2.bitwise_and(image, image, mask=mask)

    return  res

def get_colors(image, box_list):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # input is RGB image
    color_list = []
    image = edgeCorner.img_masked_rev(image)

    for i in range(0, len(box_list), 1):
        box = box_list[i]
        # location
        left = int(box[0]); top = int(box[1]); width = int(box[2]); height = int(box[3])
        # color average
        meanColor = [0, 0, 0]
        count = 0
        # for i in range(left, left + width, 1):
        #     for j in range(top, top + height, 1):
        for i in range(left, min(left + width, image.shape[1]), 1):
            for j in range(int(top+height/3), min(int(top+height*2/3), image.shape[0]), 1):
                pixel = image[j][i]
                R = int(pixel[0]); G = int(pixel[1]); B = int(pixel[2])
                if R or G or B:
                    meanColor[0] += R; meanColor[1] += G; meanColor[2] += B
                    count += 1
        if count == 0:
            print("Empty Box!")
        else:
            meanColor[0] /= count
            meanColor[1] /= count
            meanColor[2] /= count
            # 对RGB三维向量进行伸展操作，以提高聚类精度，但是可能会对人工阈值产生影响
            # meanColor[0] = meanColor[0] * 100000
            # meanColor[1] = meanColor[1] * 100000
            # meanColor[2] = meanColor[2] * 100000

        color_list.append(meanColor)

    fp=open("color.log","a+",encoding="utf-8")
    fp.write(str(color_list)+'\n')
    fp.close()

    print(color_list)
    return color_list


def get_colors_max(image, box_list):
    color_list = []
    image = edgeCorner.img_masked_rev(image)
    # cv2.imshow("masked",image)
    # cv2.waitKey(10)

    # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    for i in range(0, len(box_list), 1):
        box = box_list[i]
        # location
        left = int(box[0]); top = int(box[1]); width = int(box[2]); height = int(box[3])


        pixel_list = []
        for i in range(left, left + width, 1):
            for j in range(top, top + height, 1):
                pixel = image[j][i]
                R = int(pixel[0]); G = int(pixel[1]); B = int(pixel[2])
                if R or G or B:
                    R = R // 4 * 4
                    G = G // 4 * 4
                    B = B // 4 * 4
                    pixel_list.append([R,G,B])

        # pixel_list.sort()
        # print(pixel_list)
        maxColor = max(pixel_list, key=pixel_list.count)
        # print(maxColor)


        color_list.append(maxColor)

    fp=open("color.log","a+",encoding="utf-8")
    fp.write(str(color_list)+'\n')
    fp.close()
    print(color_list)

    return color_list

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

    def __init__(self, k=2, max_iterations=1000, varepsilon=0.0001, resetTimes=10):
        self.k = k
        self.max_iterations = max_iterations
        self.varepsilon = varepsilon
        self.resetTimes = resetTimes

    # 从所有样本中随机选取self.k样本作为初始的聚类中心
    def init_random_centroids(self, X):
        n_samples, n_features = np.shape(X)
        # print(n_samples,n_features)
        centroids = np.zeros((self.k, n_features))

        idx = random.sample(range(n_samples),self.k);
        for i in range(self.k):
            centroid = X[idx[i]]
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

    def get_cluster_labels_new(self, centroids, clusters, X):
        dist_list = []
        for centroid in centroids:
            dist = np.sqrt(np.sum(np.square(centroid)))
            dist_list.append(dist)
        # print(dist_list)
        copy = dist_list.copy()
        copy.sort()
        idx = np.zeros(len(centroids))
        for j in range(len(centroids)):
            for i in range(len(centroids)):
                if dist_list[j] == copy[i]:
                    idx[j] = i

        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            # print("cluster",cluster_i,"number:",len(cluster))
            for sample_i in cluster:
                y_pred[sample_i] = idx[cluster_i]
        return y_pred

    # 对整个数据集X进行Kmeans聚类，返回其聚类的标签
    def predict(self, X):
        times = 0
        resultGood = False
        while(times<self.resetTimes) and not resultGood:
            times = times + 1
            # 从所有样本中随机选取self.k样本作为初始的聚类中心
            centroids = self.init_random_centroids(X)

            # 迭代，直到算法收敛(上一次的聚类中心和这一次的聚类中心几乎重合)或者达到最大迭代次数
            for i in range(self.max_iterations):
                # 将所有进行归类，归类规则就是将该样本归类到与其最近的中心
                clusters = self.create_clusters(centroids, X)
                former_centroids = centroids

                # 计算新的聚类中心
                centroids = self.update_centroids(clusters, X)

                # 如果聚类中心几乎没有变化，说明算法已经收敛，退出迭代
                diff = centroids - former_centroids
                if diff.any() < self.varepsilon:
                    # print("iteration:",i)
                    break
            if(abs(len(clusters[0])-len(clusters[1])) < 4):
                resultGood = True

        return self.get_cluster_labels_new(centroids, clusters, X)


def teamClassify_kmeans(image, box_list):

    color_list = get_colors_max(image, box_list)

    res_list = []

    Clf = Kmeans(k=2)
    color_list = np.mat(color_list)
    y_pred = Clf.predict(color_list)
    # print(y_pred)

    # 每个聚类里的颜色是否已经被探知
    zero_is_detected = False
    one_is_detected = False
    two_is_detected = False

    for i in range(0, len(y_pred), 1):
        box = box_list[i]

        # add team label
        if y_pred[i] == 0:
            if not zero_is_detected:
                zero_is_detected = True
            box.append(0)
            res_list.append(box)

        elif y_pred[i] == 1:
            if not one_is_detected:
                one_is_detected = True
            box.append(1)
            res_list.append(box)

        else:
            if not two_is_detected:
                two_is_detected = True
            box.append(2)
            res_list.append(box)
    return res_list


def teamClassify(image, box_list):
    # image, box_list = input_for_classify_team()
    # 图片转化成RGB格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #print(box_list)
    res_list = []
    red_count = 0
    blue_count = 0
    rest_count = 0

    for box in box_list:
        # 位置信息
        left = int(box[0]); top = int(box[1]); width = int(box[2]); height = int(box[3])
        # 求颜色均值
        meanColor = [0, 0, 0]
        count = 0
        for i in range(left, left + width, 1):
            for j in range(top, top + height, 1):
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
        # print(meanColor)
        R = int(meanColor[0]); G = int(meanColor[1]); B = int(meanColor[2])
        res = box
        if R - G > 5 and R - B > 5:
            red_count += 1
            res.append(0)
        elif B - R > 0 and B - G > 0:
            blue_count += 1
            res.append(1)
        else:
            rest_count += 1
            res.append(2)
        #draw_1 = cv2.rectangle(image, (left, top), (left + width, top + height), (R, G, B), 2)
        res_list.append(res)
    # print("Red:", red_count, "Blue:", blue_count, "Rest:", rest_count)
    # cv2.imwrite("result.jpg", image)
    return res_list
