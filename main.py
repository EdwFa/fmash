#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import argparse

import cv2 as cv
import numpy as np
import mediapipe as mp

from collections import deque

class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument("--max_num_faces", type=int, default=1)
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    parser.add_argument('--use_brect', action='store_true')

    args = parser.parse_args()

    return args


def main():
    # Анализ аргументов #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    max_num_faces = args.max_num_faces
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = args.use_brect

    # Подготовка камеры ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Загрузка модели #############################################################
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    # Модуль измерения FPS ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    while True:
        display_fps = cvFpsCalc.get()

        # Захват камеры #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Зеркальный дисплей
        debug_image = copy.deepcopy(image)

        # Реализация обнаружения #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = face_mesh.process(image)

        # 描画 ################################################################
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                # Расчет описанного прямоугольника
                brect = calc_bounding_rect(debug_image, face_landmarks)
                # Рисование
                debug_image = draw_landmarks(debug_image, face_landmarks)
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)

        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)

        # Обработка ключей(ESC：прервать) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break

        # вывод в окно  #############################################################
        cv.imshow('FaceMesh-MediaPipe', debug_image)

    cap.release()
    cv.destroyAllWindows()


def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point.append((landmark_x, landmark_y))

        cv.circle(image, (landmark_x, landmark_y), 1, (0, 255, 0), 1)

    if len(landmark_point) > 0:
        # Справка：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg

        # левая бровь(55：Внутри、46：за пределами)
        cv.line(image, landmark_point[55], landmark_point[65], (0, 255, 0), 2)
        cv.line(image, landmark_point[65], landmark_point[52], (0, 255, 0), 2)
        cv.line(image, landmark_point[52], landmark_point[53], (0, 255, 0), 2)
        cv.line(image, landmark_point[53], landmark_point[46], (0, 255, 0), 2)

        # Правая бровь (285: внутри, 276: снаружи)
        cv.line(image, landmark_point[285], landmark_point[295], (0, 255, 0),
                2)
        cv.line(image, landmark_point[295], landmark_point[282], (0, 255, 0),
                2)
        cv.line(image, landmark_point[282], landmark_point[283], (0, 255, 0),
                2)
        cv.line(image, landmark_point[283], landmark_point[276], (0, 255, 0),
                2)

        # Левый глаз (133: внутренний угол глаза, 246: внешний угол глаза)
        cv.line(image, landmark_point[133], landmark_point[173], (0, 255, 0),
                2)
        cv.line(image, landmark_point[173], landmark_point[157], (0, 255, 0),
                2)
        cv.line(image, landmark_point[157], landmark_point[158], (0, 255, 0),
                2)
        cv.line(image, landmark_point[158], landmark_point[159], (0, 255, 0),
                2)
        cv.line(image, landmark_point[159], landmark_point[160], (0, 255, 0),
                2)
        cv.line(image, landmark_point[160], landmark_point[161], (0, 255, 0),
                2)
        cv.line(image, landmark_point[161], landmark_point[246], (0, 255, 0),
                2)

        cv.line(image, landmark_point[246], landmark_point[163], (0, 255, 0),
                2)
        cv.line(image, landmark_point[163], landmark_point[144], (0, 255, 0),
                2)
        cv.line(image, landmark_point[144], landmark_point[145], (0, 255, 0),
                2)
        cv.line(image, landmark_point[145], landmark_point[153], (0, 255, 0),
                2)
        cv.line(image, landmark_point[153], landmark_point[154], (0, 255, 0),
                2)
        cv.line(image, landmark_point[154], landmark_point[155], (0, 255, 0),
                2)
        cv.line(image, landmark_point[155], landmark_point[133], (0, 255, 0),
                2)

        # Правый глаз (362: внутренний угол глаза, 466: внешний угол глаза)
        cv.line(image, landmark_point[362], landmark_point[398], (0, 255, 0),
                2)
        cv.line(image, landmark_point[398], landmark_point[384], (0, 255, 0),
                2)
        cv.line(image, landmark_point[384], landmark_point[385], (0, 255, 0),
                2)
        cv.line(image, landmark_point[385], landmark_point[386], (0, 255, 0),
                2)
        cv.line(image, landmark_point[386], landmark_point[387], (0, 255, 0),
                2)
        cv.line(image, landmark_point[387], landmark_point[388], (0, 255, 0),
                2)
        cv.line(image, landmark_point[388], landmark_point[466], (0, 255, 0),
                2)

        cv.line(image, landmark_point[466], landmark_point[390], (0, 255, 0),
                2)
        cv.line(image, landmark_point[390], landmark_point[373], (0, 255, 0),
                2)
        cv.line(image, landmark_point[373], landmark_point[374], (0, 255, 0),
                2)
        cv.line(image, landmark_point[374], landmark_point[380], (0, 255, 0),
                2)
        cv.line(image, landmark_point[380], landmark_point[381], (0, 255, 0),
                2)
        cv.line(image, landmark_point[381], landmark_point[382], (0, 255, 0),
                2)
        cv.line(image, landmark_point[382], landmark_point[362], (0, 255, 0),
                2)

        # Рот (308: правый конец, 78: левый конец)
        cv.line(image, landmark_point[308], landmark_point[415], (0, 255, 0),
                2)
        cv.line(image, landmark_point[415], landmark_point[310], (0, 255, 0),
                2)
        cv.line(image, landmark_point[310], landmark_point[311], (0, 255, 0),
                2)
        cv.line(image, landmark_point[311], landmark_point[312], (0, 255, 0),
                2)
        cv.line(image, landmark_point[312], landmark_point[13], (0, 255, 0), 2)
        cv.line(image, landmark_point[13], landmark_point[82], (0, 255, 0), 2)
        cv.line(image, landmark_point[82], landmark_point[81], (0, 255, 0), 2)
        cv.line(image, landmark_point[81], landmark_point[80], (0, 255, 0), 2)
        cv.line(image, landmark_point[80], landmark_point[191], (0, 255, 0), 2)
        cv.line(image, landmark_point[191], landmark_point[78], (0, 255, 0), 2)

        cv.line(image, landmark_point[78], landmark_point[95], (0, 255, 0), 2)
        cv.line(image, landmark_point[95], landmark_point[88], (0, 255, 0), 2)
        cv.line(image, landmark_point[88], landmark_point[178], (0, 255, 0), 2)
        cv.line(image, landmark_point[178], landmark_point[87], (0, 255, 0), 2)
        cv.line(image, landmark_point[87], landmark_point[14], (0, 255, 0), 2)
        cv.line(image, landmark_point[14], landmark_point[317], (0, 255, 0), 2)
        cv.line(image, landmark_point[317], landmark_point[402], (0, 255, 0),
                2)
        cv.line(image, landmark_point[402], landmark_point[318], (0, 255, 0),
                2)
        cv.line(image, landmark_point[318], landmark_point[324], (0, 255, 0),
                2)
        cv.line(image, landmark_point[324], landmark_point[308], (0, 255, 0),
                2)

    return image

def draw_pose_landmarks(image, landmarks, upper_body_only, visibility_th=0.5):
	image_width, image_height = image.shape[1], image.shape[0]
	landmark_point = []
	for index, landmark in enumerate(landmarks.landmark):
		landmark_x = min(int(landmark.x * image_width), image_width - 1)
		landmark_y = min(int(landmark.y * image_height), image_height - 1)
		landmark_z = landmark.z
		landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])
		if landmark.visibility < visibility_th:
			continue
		if index == 0:  # 鼻
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 1:  # 右目：目頭
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 2:  # 右目：瞳
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 3:  # 右目：目尻
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 4:  # 左目：目頭
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 5:  # 左目：瞳
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 6:  # 左目：目尻
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 7:  # 右耳
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 8:  # 左耳
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 9:  # 口：左端
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 10:  # 口：左端
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 11:  # 右肩
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 12:  # 左肩
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 13:  # 右肘
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 14:  # 左肘
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 15:  # 右手首
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 16:  # 左手首
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 17:  # 右手1(外側端)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 18:  # 左手1(外側端)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 19:  # 右手2(先端)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 20:  # 左手2(先端)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 21:  # 右手3(内側端)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 22:  # 左手3(内側端)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 23:  # 腰(右側)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 24:  # 腰(左側)
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 25:  # 右ひざ
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 26:  # 左ひざ
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 27:  # 右足首
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 28:  # 左足首
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 29:  # 右かかと
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 30:  # 左かかと
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 31:  # 右つま先
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if index == 32:  # 左つま先
			cv.circle(image, (landmark_x, landmark_y), 5, (188, 5, 245), 2)
		if not upper_body_only:
			cv.putText(image, "z:" + str(round(landmark_z, 3)),
						(landmark_x - 10, landmark_y - 10),
						cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 22, 10), 1,
						cv.LINE_AA)
	if len(landmark_point) > 0:
		# 右目
		if landmark_point[1][0] > visibility_th and landmark_point[2][
				0] > visibility_th:
			cv.line(image, landmark_point[1][1], landmark_point[2][1],
					(188, 5, 245), 2)
		if landmark_point[2][0] > visibility_th and landmark_point[3][
				0] > visibility_th:
			cv.line(image, landmark_point[2][1], landmark_point[3][1],
					(188, 5, 245), 2)
		# 左目
		if landmark_point[4][0] > visibility_th and landmark_point[5][
				0] > visibility_th:
			cv.line(image, landmark_point[4][1], landmark_point[5][1],
					(188, 5, 245), 2)
		if landmark_point[5][0] > visibility_th and landmark_point[6][
				0] > visibility_th:
			cv.line(image, landmark_point[5][1], landmark_point[6][1],
					(188, 5, 245), 2)
		# 口
		if landmark_point[9][0] > visibility_th and landmark_point[10][
				0] > visibility_th:
			cv.line(image, landmark_point[9][1], landmark_point[10][1],
					(188, 5, 245), 2)
		# 肩
		if landmark_point[11][0] > visibility_th and landmark_point[12][
				0] > visibility_th:
			cv.line(image, landmark_point[11][1], landmark_point[12][1],
					(188, 5, 245), 2)
		# 右腕
		if landmark_point[11][0] > visibility_th and landmark_point[13][
				0] > visibility_th:
			cv.line(image, landmark_point[11][1], landmark_point[13][1],
					(188, 5, 245), 2)
		if landmark_point[13][0] > visibility_th and landmark_point[15][
				0] > visibility_th:
			cv.line(image, landmark_point[13][1], landmark_point[15][1],
					(188, 5, 245), 2)
		# 左腕
		if landmark_point[12][0] > visibility_th and landmark_point[14][
				0] > visibility_th:
			cv.line(image, landmark_point[12][1], landmark_point[14][1],
					(188, 5, 245), 2)
		if landmark_point[14][0] > visibility_th and landmark_point[16][
				0] > visibility_th:
			cv.line(image, landmark_point[14][1], landmark_point[16][1],
					(188, 5, 245), 2)
		# 右手
		if landmark_point[15][0] > visibility_th and landmark_point[17][
				0] > visibility_th:
			cv.line(image, landmark_point[15][1], landmark_point[17][1],
					(188, 5, 245), 2)
		if landmark_point[17][0] > visibility_th and landmark_point[19][
				0] > visibility_th:
			cv.line(image, landmark_point[17][1], landmark_point[19][1],
					(188, 5, 245), 2)
		if landmark_point[19][0] > visibility_th and landmark_point[21][
				0] > visibility_th:
			cv.line(image, landmark_point[19][1], landmark_point[21][1],
					(188, 5, 245), 2)
		if landmark_point[21][0] > visibility_th and landmark_point[15][
				0] > visibility_th:
			cv.line(image, landmark_point[21][1], landmark_point[15][1],
					(188, 5, 245), 2)
		# 左手
		if landmark_point[16][0] > visibility_th and landmark_point[18][
				0] > visibility_th:
			cv.line(image, landmark_point[16][1], landmark_point[18][1],
					(188, 5, 245), 2)
		if landmark_point[18][0] > visibility_th and landmark_point[20][
				0] > visibility_th:
			cv.line(image, landmark_point[18][1], landmark_point[20][1],
					(188, 5, 245), 2)
		if landmark_point[20][0] > visibility_th and landmark_point[22][
				0] > visibility_th:
			cv.line(image, landmark_point[20][1], landmark_point[22][1],
					(188, 5, 245), 2)
		if landmark_point[22][0] > visibility_th and landmark_point[16][
				0] > visibility_th:
			cv.line(image, landmark_point[22][1], landmark_point[16][1],
					(188, 5, 245), 2)
		# 胴体
		if landmark_point[11][0] > visibility_th and landmark_point[23][
				0] > visibility_th:
			cv.line(image, landmark_point[11][1], landmark_point[23][1],
					(188, 5, 245), 2)
		if landmark_point[12][0] > visibility_th and landmark_point[24][
				0] > visibility_th:
			cv.line(image, landmark_point[12][1], landmark_point[24][1],
					(188, 5, 245), 2)
		if landmark_point[23][0] > visibility_th and landmark_point[24][
				0] > visibility_th:
			cv.line(image, landmark_point[23][1], landmark_point[24][1],
					(188, 5, 245), 2)
		if len(landmark_point) > 25:
			# 右足
			if landmark_point[23][0] > visibility_th and landmark_point[25][
					0] > visibility_th:
				cv.line(image, landmark_point[23][1], landmark_point[25][1],
						(188, 5, 245), 2)
			if landmark_point[25][0] > visibility_th and landmark_point[27][
					0] > visibility_th:
				cv.line(image, landmark_point[25][1], landmark_point[27][1],
						(188, 5, 245), 2)
			if landmark_point[27][0] > visibility_th and landmark_point[29][
					0] > visibility_th:
				cv.line(image, landmark_point[27][1], landmark_point[29][1],
						(188, 5, 245), 2)
			if landmark_point[29][0] > visibility_th and landmark_point[31][
					0] > visibility_th:
				cv.line(image, landmark_point[29][1], landmark_point[31][1],
						(188, 5, 245), 2)
			# 左足
			if landmark_point[24][0] > visibility_th and landmark_point[26][
					0] > visibility_th:
				cv.line(image, landmark_point[24][1], landmark_point[26][1],
						(188, 5, 245), 2)
			if landmark_point[26][0] > visibility_th and landmark_point[28][
					0] > visibility_th:
				cv.line(image, landmark_point[26][1], landmark_point[28][1],
						(188, 5, 245), 2)
			if landmark_point[28][0] > visibility_th and landmark_point[30][
					0] > visibility_th:
				cv.line(image, landmark_point[28][1], landmark_point[30][1],
						(188, 5, 245), 2)
			if landmark_point[30][0] > visibility_th and landmark_point[32][
					0] > visibility_th:
				cv.line(image, landmark_point[30][1], landmark_point[32][1],
						(188, 5, 245), 2)
	return image




def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # описанный прямоугольник
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image


if __name__ == '__main2__':
    main()