import numpy as np
import cv2
import math

# 坐标转换核心代码如下
def cvt_pos(pos,cvt_mat_t):
    u = pos[0]
    v = pos[1]
    x = (cvt_mat_t[0][0]*u+cvt_mat_t[0][1]*v+cvt_mat_t[0][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])
    y = (cvt_mat_t[1][0]*u+cvt_mat_t[1][1]*v+cvt_mat_t[1][2])/(cvt_mat_t[2][0]*u+cvt_mat_t[2][1]*v+cvt_mat_t[2][2])

    return (int(x),int(y))

def get_points_tran(points_list, M):
    '''透视变换坐标转换'''
    for i in points_list:
        i[0],i[1] = cvt_pos([i[0],i[1]],M)
        i[2],i[3] = cvt_pos([i[2],i[3]],M)
        i[4],i[5] = cvt_pos([i[4],i[5]],M)
        i[6],i[7] = cvt_pos([i[6],i[7]],M)
    return points_list

def pointPersp(point_list, M):
    res_list = []
    for poi in point_list:
        a = float(poi[0]*M[0][0] + poi[1]*M[0][1] + M[0][2])
        b = float(poi[0]*M[2][0] + poi[1]*M[2][1] + M[2][2])
        x = a / b
        x = int(x)
        a = float(poi[0]*M[1][0] + poi[1]*M[1][1] + M[1][2])
        b = float(poi[0]*M[2][0] + poi[1]*M[2][1] + M[2][2])
        y = a / b
        y = int(y)
        res_list.append([x,y])
    return res_list


def persp(img, total_points_list):
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
    # print(M)
    out_img = cv2.warpPerspective(img,M,(w,h))

    total_points_list = pointPersp(total_points_list, M)

    return out_img, total_points_list