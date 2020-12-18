import cv2
import numpy as np


def get_contour(img):
    """获取连通域
    :param img: 输入图片
    :return: 最大连通域
    """
    ret, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        # print("轮廓 %d 的面积是:%d" % (i, area))

        areas.append(area)
    index = np.argmax(areas)
    print("轮廓 %d 的面积是:%d" % (index, cv2.contourArea(contours[index])))

    return img_bin, contours[index]


def resize_show_image(img_name, image):
    """
        缩放显示图片
    :param img_name:  显示名称
    :param image:  图片名称
    """
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


def get_warpPerspective(points, image_src):
    """
    执行透视变换
    :param points: 输入的四个角点
    :param image_src: 输入的图片
    :return: 变换后的图片
    """
    src_point = np.float32([
        [points[2][0], points[2][1]],
        [points[3][0], points[3][1]],
        [points[1][0], points[1][1]],
        [points[0][0], points[0][1]]])
    width = 1920
    height = 1080
    dst_point = np.float32([[0, 0], [width - 1, 0],
                            [0, height - 1], [width - 1, height - 1]])

    perspective_matrix = cv2.getPerspectiveTransform(src_point, dst_point)

    img_dst = cv2.warpPerspective(image_src, perspective_matrix, (width, height))
    return img_dst


def main():
    # 读取图片
    # img_white = cv2.imread("image_white.jpg", cv2.IMREAD_GRAYSCALE)
    # img_book = cv2.imread("sin0.bmp", cv2.IMREAD_GRAYSCALE)

    image = cv2.imread("field.jpg")
    #converting into hsv image
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    lower_green = np.array([40,40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    res = cv2.bitwise_and(image, image, mask=mask)
    img_white = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    img_book = image

    resize_show_image("img_book", img_book)

    # 最大的轮廓
    img_bin, contour = get_contour(img_white)

    # 处理区域mask
    mask = np.zeros(img_white.shape, np.uint8)
    mask = cv2.drawContours(mask, [contour], 0, (255, 255, 255), -1)
    # resize_show_image("img_mask", mask)

    # # 获取区域角点图片
    # img_corner = get_cornerHarris(img_white)

    # 膨胀和 mask与角点操作
    kernel = np.ones((10, 10), np.uint8)
    img_mask = cv2.dilate(mask, kernel)
    resize_show_image("img_mask", img_mask)

    res = cv2.bitwise_and(img_book, img_book, mask=img_mask)
    resize_show_image("mask_res", res)

    cv2.imwrite("mask-res.jpg",res)
    res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)

    img_corner = get_cornerHarris(res)

    img_corner = cv2.bitwise_and(img_mask, img_corner)
    resize_show_image("image_corner", img_corner)

    # 获取四个角点的中心坐标
    contours, hierarchy = cv2.findContours(img_corner, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    points = []
    for i in range(len(contours)):
        center, radius = cv2.minEnclosingCircle(contours[i])
        points.append(center)
    print(points)

    # 透视变换
    img_dst = get_warpPerspective(points, img_book)
    resize_show_image("image_dst", img_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()