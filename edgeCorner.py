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
    return img_bin, contours[index]


def resize_show_image(img_name, image):
    # Zoom to display the picture
    cv2.namedWindow(img_name, 0)
    cv2.resizeWindow(img_name, 1075, 900)
    cv2.imshow(img_name, image)


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

    # Dilation
    kernel = np.ones((10, 10), np.uint8)
    img_mask = cv2.dilate(mask, kernel)

    res = cv2.bitwise_and(image, image, mask=img_mask)
    return  res

def cal_crossPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    k1 = (y2 - y1) / (x2 - x1)
    k2 = (y3 - y4) / (x3 - x4)
    x = int(((k1 * x1 - k2 * x3) + y3 - y1) / (k1 - k2))
    y = int(k1 * (x - x1) + y1)
    return [x, y]


def edge_detect(image):
    original_img = img_masked(image)

    # canny(): edge detection
    img1 = cv2.GaussianBlur(original_img, (3, 3), 0)
    canny = cv2.Canny(img1, 50, 150)

    # padding
    image = cv2.copyMakeBorder(canny,300,300,300,300,cv2.BORDER_CONSTANT,value=[0,255,0])
    print("Origin size:", canny.shape)
    print("Expanded size:", image.shape)


    # detect point on horizontal boundaries
    detect_x1 = int(image.shape[1] / 3)
    detect_x2 = int((image.shape[1] * 2) / 3)
    # detect point on vertical boundaries
    detect_y1 = int(image.shape[0] / 2)
    detect_y2 = int((image.shape[0] * 9) / 20)
    # horizontal boundaries detection
    y1_max = 0; y1_min = image.shape[0]
    y2_max = 0; y2_min = image.shape[0]
    for i in range (0, image.shape[0], 1):
        pixel = int(image[i][detect_x1])
        if pixel == 255:
            # pixel is white
            if y1_max < i:
                y1_max = i
            if y1_min > i:
                y1_min = i
        pixel = image[i][detect_x2]
        if pixel == 255:
            if y2_max < i:
                y2_max = i
            if y2_min > i:
                y2_min = i
    # vertical boundaries detection
    x1_max = 0; x1_min = image.shape[1]
    x2_max = 0; x2_min = image.shape[1]
    for i in range (0, image.shape[1], 1):
        pixel = int(image[detect_y1][i])
        if pixel == 255:
            if x1_max < i:
                x1_max = i
            if x1_min > i:
                x1_min = i
        pixel = int(image[detect_y2][i])
        if pixel == 255:
            if x2_max < i:
                x2_max = i
            if x2_min > i:
                x2_min = i

    # Calculation of edge linear function and find the intersection point
    # (detect_x1, y1_max) - (detect_x2, y2_max) down
    # (detect_x1, y1_min) - (detect_x2, y2_min) top
    # (x1_max, detect_y1) - (x2_max, detect_y2) right
    # (x1_min, detect_y1) - (x2_min, detect_y2) left

    # rd
    rd = cal_crossPoint(detect_x1, y1_max, detect_x2, y2_max, x1_max, detect_y1, x2_max, detect_y2)
    # ld
    ld = cal_crossPoint(detect_x1, y1_max, detect_x2, y2_max, x1_min, detect_y1, x2_min, detect_y2)
    # rt
    rt = cal_crossPoint(detect_x1, y1_min, detect_x2, y2_min, x1_max, detect_y1, x2_max, detect_y2)
    # lt
    lt = cal_crossPoint(detect_x1, y1_min, detect_x2, y2_min, x1_min, detect_y1, x2_min, detect_y2)
    # print(point_1, point_2, point_3, point_4)

    return lt, rt, rd, ld# , image

def main():
    image = cv2.imread("field.jpg")

    edge_detect(image)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()