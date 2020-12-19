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

def cal_crossPoint(A1, b1, A2, b2):
    x = (b2-b1)/(A1-A2)
    y = A1*x+b1
    return [int(x), int(y)]

def horizon_edge_points(image):
    original_img = img_masked(image)

    # canny(): edge detection
    img1 = cv2.GaussianBlur(original_img, (3, 3), 0)
    canny = cv2.Canny(img1, 50, 150)

    # padding
    image = cv2.copyMakeBorder(canny,300,300,300,300,cv2.BORDER_CONSTANT,value=[0,255,0])

    top_list = []
    down_list = []
    # detect point on horizontal boundaries
    x1 = int(image.shape[1] / 3)
    x2 = int((image.shape[1] * 2) / 3)
    for i in range(x1, x2):
        y_max = 0; y_min = image.shape[0]
        for j in range(0,image.shape[0]):
            pixel = int(image[j][i])
            if pixel == 255:
                # pixel is white
                if y_max < j:
                    y_max = j
                if y_min > j:
                    y_min = j
        top_list.append([i,y_min])
        down_list.append([i,y_max])

    return top_list, down_list


def vertical_edge_points(image, top, down):
    original_img = img_masked(image)

    # canny(): edge detection
    img1 = cv2.GaussianBlur(original_img, (3, 3), 0)
    canny = cv2.Canny(img1, 50, 150)

    # padding
    image = cv2.copyMakeBorder(canny,300,300,300,300,cv2.BORDER_CONSTANT,value=[0,255,0])

    left_list = []
    right_list = []
    # detect point on horizontal boundaries
    # print(top, down)
    y1 = int(top+(down-top)/3)
    y2 = int(top+(down-top)/2)
    # print(y1, y2)
    for i in range(y1, y2):
        x_max = 0; x_min = image.shape[1]
        for j in range(0,image.shape[1]):
            pixel = int(image[i][j])
            if pixel == 255:
                # pixel is white
                if x_max < j:
                    x_max = j
                if x_min > j:
                    x_min = j
        left_list.append([x_min, i])
        right_list.append([x_max, i])

    # print(left_list)
    return left_list, right_list


def linear_fit(point_list):
    n = len(point_list)
    # Calculate coefficients for normal equations based on dataset
    u = 0
    v = 0
    w = 0
    z = 0
    for poi in point_list:
        u = u + poi[0]
        v = v + poi[1]
        w = w + poi[0]**2
        z = z + poi[0]*poi[1]

    # Set up normal equations in matrix form
    A = np.array([[n, u], [u, w]])
    b = np.array([[v, z]]).transpose()

    x = np.linalg.solve(A,b)
    # print(x)
    return x


def corner_detect(image):
    top_list, down_list = horizon_edge_points(image)

    top = linear_fit(top_list)
    down = linear_fit(down_list)

    left_list, right_list = vertical_edge_points(image, top[0], down[0])

    left = linear_fit(left_list)
    right = linear_fit(right_list)

    # rd
    rd = cal_crossPoint(right[1], right[0], down[1], down[0])
    # ld
    ld = cal_crossPoint(left[1], left[0], down[1], down[0])
    # rt
    rt = cal_crossPoint(right[1], right[0], top[1], top[0])
    # lt
    lt = cal_crossPoint(left[1], left[0], top[1], top[0])
    # print(point_1, point_2, point_3, point_4)
    # print(lt, rt, rd, ld)
    return lt, rt, rd, ld# , image

def main():
    image = cv2.imread("field.jpg")

    corner_detect(image)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()