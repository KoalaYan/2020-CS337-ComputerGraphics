import cv2
import numpy as np


def get_contour(img):
    # Get connected domain
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        areas.append(area)

    index = np.argmax(areas)
    # print("轮廓 %d 的面积是:%d" % (index, cv2.contourArea(contours[index])))
    return img_bin, contours[index]


def resize_show_image(img_name, image):
    # Zoom to display the picture
    cv2.namedWindow(img_name, 0)
    cv2.resizeWindow(img_name, 1075, 900)
    cv2.imshow(img_name, image)


def get_cornerHarris(img_src):
    """
        获取图像角点
    :param img_src: 处理图像
    :return: 角点图像
    """
    img_corner = np.zeros(img_src.shape, np.uint8)
    img_gray = img_src.copy()
    img_gray = np.float32(img_gray)

    img_dist = cv2.cornerHarris(img_gray, 5, 5, 0.04)
    img_dist = cv2.dilate(img_dist, None)

    img_corner[img_dist > 0.01 * img_dist.max()] = [255]

    return img_corner


def img_masked(image):
    #converting into hsv image
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_green = np.array([40,40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)

    img_white = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    # The largest fragment
    img_bin, contour = get_contour(img_white)

    mask = np.zeros(img_white.shape, np.uint8)
    mask = cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)

    # 获取区域角点图片
    # img_corner = get_cornerHarris(img_white)

    # 膨胀和 mask与角点操作
    kernel = np.ones((10, 10), np.uint8)
    img_mask = cv2.dilate(mask, kernel)
    # resize_show_image("img_mask", img_mask)

    res = cv2.bitwise_and(image, image, mask=img_mask)
    # resize_show_image("mask_res", res)
    # cv2.imwrite("mask-res.jpg",res)
    return  res

def main():

    image = cv2.imread("field.jpg")

    res = img_masked(image)
    resize_show_image("mask_res", res)
    cv2.imwrite("mask-res.jpg",res)

    # res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    #
    # img_corner = get_cornerHarris(res)
    #
    # img_corner = cv2.bitwise_and(img_mask, img_corner)
    # resize_show_image("image_corner", img_corner)
    #
    # # 获取四个角点的中心坐标
    # contours, hierarchy = cv2.findContours(img_corner, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    #
    # points = []
    # for i in range(len(contours)):
    #     center, radius = cv2.minEnclosingCircle(contours[i])
    #     points.append(center)
    # print(points)
    #
    # # 透视变换
    # img_dst = get_warpPerspective(points, img_book)
    # resize_show_image("image_dst", img_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()