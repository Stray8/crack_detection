import math
import random
import cv2
import numpy as np
from numpy.ma import cos, sin
import time
from matplotlib import pyplot as plt


# 读取图片
frame = cv2.imread("crack.jfif")


# 限制图片大小
def resize_image(image, max_width):
    rows, cols = image.shape[:2]
    if cols > max_width:
        change_rate = max_width / cols
        img = cv2.resize(image, (max_width, int(cols * change_rate)), interpolation=cv2.INTER_AREA)
    if rows > max_width:
        change_rate = max_width / rows
        img = cv2.resize(image, (max_width, int(rows * change_rate)), interpolation=cv2.INTER_AREA)
    return img


# 鼠标点击事件
def mouse_click(event, x, y, flags, para):
    if event == cv2.EVENT_LBUTTONDOWN:
        print([x, y])


# 图像预处理
def image_process(image):
    # 转为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 边缘检测 得到二值化后图
    canny = cv2.Canny(gray, 50, 150)
    # 形态学变换
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilate = cv2.dilate(canny, kernel=kernel)
    dst = cv2.morphologyEx(dilate, cv2.MORPH_CLOSE, kernel=kernel, anchor=(-1, -1), iterations=3)
    result_image = cv2.morphologyEx(dst, cv2.MORPH_CLOSE, kernel=kernel)
    # retval, binary = cv2.threshold(result_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("gray", gray)
    cv2.imshow("canny", canny)
    cv2.imshow("dilate", dilate)
    return result_image


def crack_detector(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cont_list = []
    for contour in contours:
        if cv2.contourArea(contour) > 150:
            cont_list.append(contour)
    expansion_circle_list = []
    for i, cont in enumerate(cont_list):
        cv2.drawContours(frame, cont_list, i, (0, 0, 255))
        cnt = cont_list[i]
        M = cv2.moments(cnt)
        # 求中点坐标
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        # 求长度
        length = cv2.arcLength(cont, True)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
        cv2.putText(frame, str((cx, cy)), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0))
        cv2.putText(frame, str(length), cont[0][0], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
        cv2.drawContours(frame, cont_list, i, (0, 0, 255))

        # 求裂纹宽度
        # 计算轮廓最大内切圆
        left_x = min(cnt[:, 0, 0])
        right_x = max(cnt[:, 0, 0])
        down_y = max(cnt[:, 0, 1])
        up_y = min(cnt[:, 0, 1])
        upper_r = min(right_x - left_x, down_y - up_y) / 2
        precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
        Nx = 2 ** 8
        Ny = 2 ** 8
        pixel_X = np.linspace(left_x, right_x)
        pixel_Y = np.linspace(up_y, down_y)
        xx, yy = np.meshgrid(pixel_X, pixel_Y)
        in_list = []
        for i in range(pixel_X.shape[0]):
            for j in range(pixel_Y.shape[0]):
                if cv2.pointPolygonTest(cnt, (xx[i][j], yy[i][j]), False) > 0:
                    in_list.append((xx[i][j], yy[i][j]))
        in_point = np.array(in_list)
        N = len(in_point)
        rand_index = random.sample(range(N), N // 100)
        rand_index.sort()
        radius = 0
        big_r = upper_r  # 裂缝内切圆的半径最大不超过此距离
        center = None
        for id in rand_index:
            tr = iterated_optimal_incircle_radius_get(cnt, in_point[id][0], in_point[id][1], radius, big_r, precision)
            if tr > radius:
                radius = tr
                center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
        # 循环搜索剩余像素对应内切圆半径
        loops_index = [i for i in range(N) if i not in rand_index]
        for id in loops_index:
            tr = iterated_optimal_incircle_radius_get(cnt, in_point[id][0], in_point[id][1], radius, big_r, precision)
            if tr > radius:
                radius = tr
                center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变

        expansion_circle_list.append([radius, center])  # 保存每条裂缝最大内切圆的半径和圆心
        # 输出裂缝的最大宽度
        print('裂缝宽度：', round(radius * 2, 2))

    print('---------------')
    expansion_circle_radius_list = [i[0] for i in expansion_circle_list]  # 每条裂缝最大内切圆半径列表
    max_radius = max(expansion_circle_radius_list)
    max_center = expansion_circle_list[expansion_circle_radius_list.index(max_radius)][1]
    print('最大宽度：', round(max_radius * 2, 2))
    # 绘制裂缝轮廓最大内切圆
    for expansion_circle in expansion_circle_list:
        radius_s = expansion_circle[0]
        center_s = expansion_circle[1]
        if radius_s == max_radius:  # 最大内切圆，用蓝色标注
            cv2.circle(frame, (int(max_center[0]), int(max_center[1])), int(max_radius), (255, 0, 255), 2)
            cv2.putText(frame, str(int(max_radius)), (int(max_center[0]), int(max_center[1])), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0))
        else:  # 其他内切圆，用青色标注
            cv2.circle(frame, (int(center_s[0]), int(center_s[1])), int(radius_s), (255, 245, 0), 2)

    cv2.imshow("crack", frame)


def iterated_optimal_incircle_radius_get(contous, pixelx, pixely, small_r, big_r, precision):
    """
    计算轮廓内最大内切圆的半径
    Args:
        contous: 轮廓像素点array数组
        pixelx: 圆心x像素坐标
        pixely: 圆心y像素坐标
        small_r: 之前所有计算所求得的内切圆的最大半径，作为下次计算时的最小半径输入，只有半径变大时才允许位置变更，否则保持之前位置不变
        big_r: 圆的半径最大不超过此距离
        precision: 相切二分精度，采用二分法寻找最大半径

    Returns: 轮廓内切圆的半径
    """
    radius = small_r
    L = np.linspace(0, 2 * math.pi, 360)  # 确定圆散点剖分数360, 720
    circle_X = pixelx + radius * cos(L)
    circle_Y = pixely + radius * sin(L)
    for i in range(len(circle_Y)):
        if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
            return 0
    while big_r - small_r >= precision:  # 二分法寻找最大半径
        half_r = (small_r + big_r) / 2
        circle_X = pixelx + half_r * cos(L)
        circle_Y = pixely + half_r * sin(L)
        if_out = False
        for i in range(len(circle_Y)):
            if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
                big_r = half_r
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    return radius


def camera_read():
    cap = cv2.VideoCapture('crack.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("camera is none")
            break

        result = image_process(frame)
        contours, hierarchy = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cont_list = []
        for contour in contours:
            if cv2.contourArea(contour) > 60:
                cont_list.append(contour)

        expansion_circle_list = []
        for i, cont in enumerate(cont_list):
            cnt = cont_list[i]
            M = cv2.moments(cnt)
            # 求中点坐标
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # 求长度
            length = cv2.arcLength(cont, True)

            cv2.putText(frame, 'length:'+str(int(length)), (cx, cy-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # 求裂纹中心位置
            cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            cv2.putText(frame, 'center:'+str((cx, cy)), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
            # 绘制裂纹
            cv2.drawContours(frame, cont_list, i, (0, 0, 255))

            # 求裂纹宽度
            # 计算轮廓最大内切圆
            left_x = min(cnt[:, 0, 0])
            right_x = max(cnt[:, 0, 0])
            down_y = max(cnt[:, 0, 1])
            up_y = min(cnt[:, 0, 1])
            upper_r = min(right_x-left_x, down_y-up_y) / 2
            precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
            Nx = 2 ** 8
            Ny = 2 ** 8
            pixel_X = np.linspace(left_x, right_x)
            pixel_Y = np.linspace(up_y, down_y)
            xx, yy = np.meshgrid(pixel_X, pixel_Y)
            in_list = []
            for i in range(pixel_X.shape[0]):
                for j in range(pixel_Y.shape[0]):
                    if cv2.pointPolygonTest(cnt, (xx[i][j], yy[i][j]), False) > 0:
                        in_list.append((xx[i][j], yy[i][j]))
            in_point = np.array(in_list)
            N = len(in_point)
            rand_index = random.sample(range(N), N // 100)
            rand_index.sort()
            radius = 0
            big_r = upper_r  # 裂缝内切圆的半径最大不超过此距离
            center = None
            for id in rand_index:
                tr = iterated_optimal_incircle_radius_get(cnt, in_point[id][0], in_point[id][1], radius, big_r, precision)
                if tr > radius:
                    radius = tr
                    center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
            # 循环搜索剩余像素对应内切圆半径
            loops_index = [i for i in range(N) if i not in rand_index]
            for id in loops_index:
                tr = iterated_optimal_incircle_radius_get(cnt, in_point[id][0], in_point[id][1], radius, big_r, precision)
                if tr > radius:
                    radius = tr
                    center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变

            expansion_circle_list.append([radius, center])  # 保存每条裂缝最大内切圆的半径和圆心
            # 输出裂缝的最大宽度
            print('裂缝宽度：', round(radius * 2, 2))

        print('---------------')
        expansion_circle_radius_list = [i[0] for i in expansion_circle_list]  # 每条裂缝最大内切圆半径列表
        max_radius = max(expansion_circle_radius_list)
        max_center = expansion_circle_list[expansion_circle_radius_list.index(max_radius)][1]
        print('最大宽度：', round(max_radius * 2, 2))
        # 绘制裂缝轮廓最大内切圆
        for expansion_circle in expansion_circle_list:
            radius_s = expansion_circle[0]
            center_s = expansion_circle[1]
            print(center_s)
            if radius_s == max_radius:  # 最大内切圆，用蓝色标注
                cv2.circle(frame, (int(max_center[0]), int(max_center[1])), int(max_radius), (255, 0, 255), 2)
                cv2.putText(frame, 'width:'+str(int(max_radius)), (int(max_center[0]), int(max_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))
            else:  # 其他内切圆，用青色标注
                cv2.circle(frame, (int(center_s[0]), int(center_s[1])), int(radius_s), (255, 245, 0), 2)
                # cv2.putText(frame, str(int(max_radius)), (int(max_center[0]), int(max_center[1])),
                # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        cv2.imshow("img", frame)
        cv2.imshow("result", result)

        k = cv2.waitKey(0)
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


def image_read():
    # resize = resize_image(frame, 1000)

    result = image_process(frame)
    crack_detector(result)

    # cv2.imshow("frame", frame)
    cv2.imshow("result", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    camera_read()
