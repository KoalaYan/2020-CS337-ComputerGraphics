# import cv2
# import numpy as np
#
# def draw_contours(img, cnts):  # conts = contours
#     img = np.copy(img)
#     img = cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
#     return img
#
#
# def draw_min_rect_circle(img, cnts):  # conts = contours
#     img = np.copy(img)
#
#     for cnt in cnts:
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)  # blue
#
#         min_rect = cv2.minAreaRect(cnt)  # min_area_rectangle
#         min_rect = np.int0(cv2.boxPoints(min_rect))
#         cv2.drawContours(img, [min_rect], 0, (0, 255, 0), 2)  # green
#
#         (x, y), radius = cv2.minEnclosingCircle(cnt)
#         center, radius = (int(x), int(y)), int(radius)  # center and radius of minimum enclosing circle
#         img = cv2.circle(img, center, radius, (0, 0, 255), 2)  # red
#     return img
#
#
# def draw_approx_hull_polygon(img, cnts):
#     # img = np.copy(img)
#     img = np.zeros(img.shape, dtype=np.uint8)
#
#     cv2.drawContours(img, cnts, -1, (255, 0, 0), 2)  # blue
#
#     min_side_len = img.shape[0] / 32  # 多边形边长的最小值 the minimum side length of polygon
#     min_poly_len = img.shape[0] / 16  # 多边形周长的最小值 the minimum round length of polygon
#     min_side_num = 3  # 多边形边数的最小值
#     approxs = [cv2.approxPolyDP(cnt, min_side_len, True) for cnt in cnts]  # 以最小边长为限制画出多边形
#     approxs = [approx for approx in approxs if cv2.arcLength(approx, True) > min_poly_len]  # 筛选出周长大于 min_poly_len 的多边形
#     approxs = [approx for approx in approxs if len(approx) > min_side_num]  # 筛选出边长数大于 min_side_num 的多边形
#     # Above codes are written separately for the convenience of presentation.
#     cv2.polylines(img, approxs, True, (0, 255, 0), 2)  # green
#
#     hulls = [cv2.convexHull(cnt) for cnt in cnts]
#     cv2.polylines(img, hulls, True, (0, 0, 255), 2)  # red
#
#     # for cnt in cnts:
#     #     cv2.drawContours(img, [cnt, ], -1, (255, 0, 0), 2)  # blue
#     #
#     #     epsilon = 0.02 * cv2.arcLength(cnt, True)
#     #     approx = cv2.approxPolyDP(cnt, epsilon, True)
#     #     cv2.polylines(img, [approx, ], True, (0, 255, 0), 2)  # green
#     #
#     #     hull = cv2.convexHull(cnt)
#     #     cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
#     return img
#
#
# def run():
#     image = cv2.imread('field.png')  # a black objects on white image is better
#
#     # gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
#     # ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
#     thresh = cv2.Canny(image, 128, 256)
#
#     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     # print(hierarchy, ":hierarchy")
#     """
#     [[[-1 -1 -1 -1]]] :hierarchy  # cv2.Canny()
#
#     [[[ 1 -1 -1 -1]
#       [ 2  0 -1 -1]
#       [ 3  1 -1 -1]
#       [-1  2 -1 -1]]] :hierarchy  # cv2.threshold()
#     """
#
#     imgs = [
#         image, thresh,
#         draw_min_rect_circle(image, contours),
#         draw_approx_hull_polygon(image, contours),
#     ]
#
#     for img in imgs:
#         cv2.imwrite("%s.jpg" % id(img), img)
#         cv2.imshow("contours", img)
#         cv2.waitKey(1943)
#
# if __name__ == '__main__':
#     run()

import cv2
import numpy as np
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        print(xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (0,0,0), thickness = 1)
        cv2.imshow("image", img)


if __name__ == '__main__':
    img = cv2.imread("field.jpg")

    # cv2.namedWindow("image")
    # cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
    # cv2.imshow("image", img)
    #
    # while(True):
    #     try:
    #         cv2.waitKey(100)
    #     except Exception:
    #         cv2.destroyWindow("image")
    #         break
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindow()

# 323,398
# 1593,408
# 1910,693
# 0,675
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

    cv2.imwrite("persp.jpg", out_img)

    cv2.waitKey(0)#等待键盘输入，不输入 则无限等待
    cv2.destroyAllWindows()#清除所以窗口