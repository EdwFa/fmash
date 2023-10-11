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

RADIUS = 5
THIKNESS = 2
RED = 240
GREEN = 40
BLUE = 240


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)
    parser.add_argument('--upper_body_only', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='face mesh min_detection_confidence',
                        type=float,
                        default=0.5)
    parser.add_argument("--min_tracking_confidence",
                        help='face mesh min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument('--use_brect', action='store_true')
    args = parser.parse_args()
    return args


def holistic():
    # Анализ аргументов #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    upper_body_only = args.upper_body_only
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    use_brect = args.use_brect
    # Подготовка камеры ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Загрузка модели #############################################################
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
       # upper_body_only=upper_body_only,
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
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        # Лицо  ###########################################################
        face_landmarks = results.face_landmarks
        if face_landmarks is not None:
            # Расчет описанного прямоугольника
            brect = calc_bounding_rect(debug_image, face_landmarks)
            # Рисование
            debug_image = draw_face_landmarks(debug_image, face_landmarks)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        # Поза ###############################################################
        pose_landmarks = results.pose_landmarks
        if pose_landmarks is not None:
            # Расчет описанного прямоугольника
            brect = calc_bounding_rect(debug_image, pose_landmarks)
            # Рисование
            debug_image = draw_pose_landmarks(debug_image, pose_landmarks,
                                              upper_body_only)
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        # Руки ###############################################################
        left_hand_landmarks = results.left_hand_landmarks
        right_hand_landmarks = results.right_hand_landmarks
        # левая рука
        if left_hand_landmarks is not None:
            # Расчет центра тяжести ладони
            cx, cy = calc_palm_moment(debug_image, left_hand_landmarks)
            # Расчет описанного прямоугольника
            brect = calc_bounding_rect(debug_image, left_hand_landmarks)
            # Рисование
            debug_image = draw_hands_landmarks(debug_image, cx, cy,
                                               left_hand_landmarks,
                                               upper_body_only, 'R')
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        # правая рука
        if right_hand_landmarks is not None:
            # Расчет центра тяжести ладони
            cx, cy = calc_palm_moment(debug_image, right_hand_landmarks)
            # Расчет описанного прямоугольника
            brect = calc_bounding_rect(debug_image, right_hand_landmarks)
            # Рисование
            debug_image = draw_hands_landmarks(debug_image, cx, cy,
                                               right_hand_landmarks,
                                               upper_body_only, 'L')
            debug_image = draw_bounding_rect(use_brect, debug_image, brect)
        cv.putText(debug_image, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2, cv.LINE_AA)
        # drawing_utils образец #############################################################
        # mp_drawing = mp.solutions.drawing_utils
        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holis)

        # Обработка ключей (ESC: завершить) #################################################
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        # отражение экрана #############################################################
        cv.imshow('Парсинг движений человека', debug_image)
    cap.release()
    cv.destroyAllWindows()


def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    palm_array = np.empty((0, 2), int)
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        if index == 0:  # Запястье 1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # Запястье 2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # Указательный палец: корень
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # Средний палец: корень
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # Безымянный палец: корень
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # Мизинец: основание
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    return cx, cy


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


def draw_hands_landmarks(image,
                         cx,
                         cy,
                         landmarks,
                         upper_body_only,
                         handedness_str='R'):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    # Ключевой момент
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append((landmark_x, landmark_y))
        if index == 0:  # Запястье 1
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 1:  # Запястье 2
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 2:  # Большой палец: корень
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 3:  # Большой палец: 1-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 4:  # Большой палец: кончик пальца
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
            cv.circle(image, (landmark_x, landmark_y), 12, (RED, GREEN, BLUE), THIKNESS)
        if index == 5:  # Указательный палец: корень
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 6:  # Указательный палец: 2-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 7:  # Указательный палец: 1-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 8:  # Указательный палец: кончик пальца
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
            cv.circle(image, (landmark_x, landmark_y), 12, (RED, GREEN, BLUE), THIKNESS)
        if index == 9:  # Средний палец: корень
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 10:  # Средний палец: 2-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 11:  # Средний палец: 1-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 12:  # Средний палец: кончик пальца
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
            cv.circle(image, (landmark_x, landmark_y), 12, (RED, GREEN, BLUE), THIKNESS)
        if index == 13:  # Безымянный палец: корень
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 14:  # Безымянный палец: 2-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 15:  # Безымянный палец: 1-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 16:  # Безымянный палец: кончик пальца
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
            cv.circle(image, (landmark_x, landmark_y), 12, (RED, GREEN, BLUE), THIKNESS)
        if index == 17:  # Мизинец: основание
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 18:  # Мизинец: 2-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 19:  # Мизинец: 1-й сустав
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 20:  # мизинец: палец вперед
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
            cv.circle(image, (landmark_x, landmark_y), 12, (RED, GREEN, BLUE), THIKNESS)
        if not upper_body_only:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 22, 10), 1,
                       cv.LINE_AA)
    # Линия соединения
    if len(landmark_point) > 0:
        # большой палец
        cv.line(image, landmark_point[2], landmark_point[3], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[3], landmark_point[4], (RED, GREEN, BLUE), THIKNESS)
        # указательный палец
        cv.line(image, landmark_point[5], landmark_point[6], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[6], landmark_point[7], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[7], landmark_point[8], (RED, GREEN, BLUE), THIKNESS)
        # средний палец
        cv.line(image, landmark_point[9], landmark_point[10], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[10], landmark_point[11], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[11], landmark_point[12], (RED, GREEN, BLUE), THIKNESS)
        # Безымянный палец
        cv.line(image, landmark_point[13], landmark_point[14], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[14], landmark_point[15], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[15], landmark_point[16], (RED, GREEN, BLUE), THIKNESS)
        # мизинец
        cv.line(image, landmark_point[17], landmark_point[18], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[18], landmark_point[19], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[19], landmark_point[20], (RED, GREEN, BLUE), THIKNESS)
        # Ладонь
        cv.line(image, landmark_point[0], landmark_point[1], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[1], landmark_point[2], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[2], landmark_point[5], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[5], landmark_point[9], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[9], landmark_point[13], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[13], landmark_point[17], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[17], landmark_point[0], (RED, GREEN, BLUE), THIKNESS)
    # Центр тяжести + слева и справа
    if len(landmark_point) > 0:
        cv.circle(image, (cx, cy), 12, (80, 44, 110), 2)
        cv.putText(image, handedness_str, (cx - 6, cy + 6),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (80, 44, 110), 2, cv.LINE_AA)
    return image


def draw_face_landmarks(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append((landmark_x, landmark_y))
        cv.circle(image, (landmark_x, landmark_y), 1, (RED, GREEN, BLUE), THIKNESS)
    if len(landmark_point) > 0:
        # Справка：https://github.com/tensorflow/tfjs-models/blob/master/facemesh/mesh_map.jpg
        # Левая бровь (55: внутри, 46: снаружи)
        cv.line(image, landmark_point[55], landmark_point[65], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[65], landmark_point[52], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[52], landmark_point[53], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[53], landmark_point[46], (RED, GREEN, BLUE), THIKNESS)
        # Правая бровь (285: внутри, 276: снаружи)
        cv.line(image, landmark_point[285], landmark_point[295], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[295], landmark_point[282], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[282], landmark_point[283], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[283], landmark_point[276], (RED, GREEN, BLUE), THIKNESS)
        # Левый глаз (133: внутренний угол глаза, 246: внешний угол глаза)
        cv.line(image, landmark_point[133], landmark_point[173], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[173], landmark_point[157], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[157], landmark_point[158], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[158], landmark_point[159], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[159], landmark_point[160], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[160], landmark_point[161], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[161], landmark_point[246], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[246], landmark_point[163], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[163], landmark_point[144], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[144], landmark_point[145], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[145], landmark_point[153], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[153], landmark_point[154], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[154], landmark_point[155], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[155], landmark_point[133], (RED, GREEN, BLUE), THIKNESS)
        # Правый глаз (362: внутренний угол глаза, 466: внешний угол глаза)
        cv.line(image, landmark_point[362], landmark_point[398], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[398], landmark_point[384], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[384], landmark_point[385], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[385], landmark_point[386], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[386], landmark_point[387], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[387], landmark_point[388], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[388], landmark_point[466], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[466], landmark_point[390], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[390], landmark_point[373], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[373], landmark_point[374], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[374], landmark_point[380], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[380], landmark_point[381], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[381], landmark_point[382], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[382], landmark_point[362], (RED, GREEN, BLUE), THIKNESS)
        # Рот (308: правый конец, 78: левый конец)
        cv.line(image, landmark_point[308], landmark_point[415], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[415], landmark_point[310], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[310], landmark_point[311], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[311], landmark_point[312], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[312], landmark_point[13], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[13], landmark_point[82], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[82], landmark_point[81], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[81], landmark_point[80], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[80], landmark_point[191], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[191], landmark_point[78], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[78], landmark_point[95], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[95], landmark_point[88], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[88], landmark_point[178], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[178], landmark_point[87], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[87], landmark_point[14], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[14], landmark_point[317], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[317], landmark_point[402], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[402], landmark_point[318], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[318], landmark_point[324], (RED, GREEN, BLUE), THIKNESS)
        cv.line(image, landmark_point[324], landmark_point[308], (RED, GREEN, BLUE), THIKNESS)
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
        if index == 0:  # нос
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 1:  # Правый глаз: внутренний угол глаза
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 2:  # Правый глаз: зрачок
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 3:  # Правый глаз: внешний угол глаза
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 4:  # Левый глаз: внутренний угол глаза
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 5:  # Левый глаз: зрачок
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 6:  # Левый глаз: внешний угол глаза
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 7:  # Правое ухо
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 8:  # Левое ухо
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 9:  # Рот: левый конец
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 10:  # Рот: правый конец
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 11:  # правое плечо
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 12:  # левое плечо
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 13:  # Правый локоть
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 14:  # Левый локоть
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 15:  # правая рука
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 16:  # Левое запястье
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 17:  # Правая рука 1 (внешний конец)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 18:  # Левая рука 1 (внешний конец)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 19:  # Правая рука 2 (вершина)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 20:  # Левая рука 2 (наконечник)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 21:  # Правая рука 3 (внутренний край)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 22:  # Левая рука 3 (внутренний конец)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 23:  # талия (правая сторона)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 24:  # талия (левая сторона)
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 25:  # Правое колено
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 26:  # 左ひざ
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 27:  # 右足首
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 28:  # 左足首
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 29:  # 右かかと
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 30:  # 左かかと
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 31:  # 右つま先
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if index == 32:  # 左つま先
            cv.circle(image, (landmark_x, landmark_y), RADIUS, (RED, GREEN, BLUE), THIKNESS)
        if not upper_body_only:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),
                       (landmark_x - 10, landmark_y - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (80, 22, 10), 1,
                       cv.LINE_AA)
    if len(landmark_point) > 0:
        # 右目
        if landmark_point[1][0] > visibility_th and landmark_point[2][0] > visibility_th:
            cv.line(image, landmark_point[1][1], landmark_point[2][1], (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[2][0] > visibility_th and landmark_point[3][0] > visibility_th:
            cv.line(image, landmark_point[2][1], landmark_point[3][1], (RED, GREEN, BLUE), THIKNESS)
        # левый глаз
        if landmark_point[4][0] > visibility_th and landmark_point[5][0] > visibility_th:
            cv.line(image, landmark_point[4][1], landmark_point[5][1], (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[5][0] > visibility_th and landmark_point[6][0] > visibility_th:
            cv.line(image, landmark_point[5][1], landmark_point[6][1], (RED, GREEN, BLUE), THIKNESS)
        # рот
        if landmark_point[9][0] > visibility_th and landmark_point[10][0] > visibility_th:
            cv.line(image, landmark_point[9][1], landmark_point[10][1], (RED, GREEN, BLUE), THIKNESS)
        # 肩
        if landmark_point[11][0] > visibility_th and landmark_point[12][0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[12][1], (RED, GREEN, BLUE), THIKNESS)
        # 右腕
        if landmark_point[11][0] > visibility_th and landmark_point[13][0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[13][1], (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[13][0] > visibility_th and landmark_point[15][0] > visibility_th:
            cv.line(image, landmark_point[13][1], landmark_point[15][1], (RED, GREEN, BLUE), THIKNESS)
        # 左腕
        if landmark_point[12][0] > visibility_th and landmark_point[14][0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[14][1], (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[14][0] > visibility_th and landmark_point[16][0] > visibility_th:
            cv.line(image, landmark_point[14][1], landmark_point[16][1], (RED, GREEN, BLUE), THIKNESS)
        # 右手
        if landmark_point[15][0] > visibility_th and landmark_point[17][0] > visibility_th:
            cv.line(image, landmark_point[15][1], landmark_point[17][1], (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[17][0] > visibility_th and landmark_point[19][0] > visibility_th:
            cv.line(image, landmark_point[17][1], landmark_point[19][1], (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[19][0] > visibility_th and landmark_point[21][0] > visibility_th:
            cv.line(image, landmark_point[19][1], landmark_point[21][1], (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[21][0] > visibility_th and landmark_point[15][0] > visibility_th:
            cv.line(image, landmark_point[21][1], landmark_point[15][1], (RED, GREEN, BLUE), THIKNESS)
        # 左手
        if landmark_point[16][0] > visibility_th and landmark_point[18][
            0] > visibility_th:
            cv.line(image, landmark_point[16][1], landmark_point[18][1],
                    (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[18][0] > visibility_th and landmark_point[20][
            0] > visibility_th:
            cv.line(image, landmark_point[18][1], landmark_point[20][1],
                    (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[20][0] > visibility_th and landmark_point[22][
            0] > visibility_th:
            cv.line(image, landmark_point[20][1], landmark_point[22][1],
                    (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[22][0] > visibility_th and landmark_point[16][
            0] > visibility_th:
            cv.line(image, landmark_point[22][1], landmark_point[16][1],
                    (RED, GREEN, BLUE), THIKNESS)
        # 胴体
        if landmark_point[11][0] > visibility_th and landmark_point[23][
            0] > visibility_th:
            cv.line(image, landmark_point[11][1], landmark_point[23][1],
                    (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[12][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
            cv.line(image, landmark_point[12][1], landmark_point[24][1],
                    (RED, GREEN, BLUE), THIKNESS)
        if landmark_point[23][0] > visibility_th and landmark_point[24][
            0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[24][1],
                    (RED, GREEN, BLUE), THIKNESS)
        if len(landmark_point) > 25:
            # 右足
            if landmark_point[23][0] > visibility_th and landmark_point[25][
                0] > visibility_th:
                cv.line(image, landmark_point[23][1], landmark_point[25][1],
                        (RED, GREEN, BLUE), THIKNESS)
            if landmark_point[25][0] > visibility_th and landmark_point[27][
                0] > visibility_th:
                cv.line(image, landmark_point[25][1], landmark_point[27][1],
                        (RED, GREEN, BLUE), THIKNESS)
            if landmark_point[27][0] > visibility_th and landmark_point[29][
                0] > visibility_th:
                cv.line(image, landmark_point[27][1], landmark_point[29][1],
                        (RED, GREEN, BLUE), THIKNESS)
            if landmark_point[29][0] > visibility_th and landmark_point[31][
                0] > visibility_th:
                cv.line(image, landmark_point[29][1], landmark_point[31][1],
                        (RED, GREEN, BLUE), THIKNESS)
            # 左足
            if landmark_point[24][0] > visibility_th and landmark_point[26][
                0] > visibility_th:
                cv.line(image, landmark_point[24][1], landmark_point[26][1],
                        (RED, GREEN, BLUE), THIKNESS)
            if landmark_point[26][0] > visibility_th and landmark_point[28][
                0] > visibility_th:
                cv.line(image, landmark_point[26][1], landmark_point[28][1],
                        (RED, GREEN, BLUE), THIKNESS)
            if landmark_point[28][0] > visibility_th and landmark_point[30][
                0] > visibility_th:
                cv.line(image, landmark_point[28][1], landmark_point[30][1],
                        (RED, GREEN, BLUE), THIKNESS)
            if landmark_point[30][0] > visibility_th and landmark_point[32][
                0] > visibility_th:
                cv.line(image, landmark_point[30][1], landmark_point[32][1],
                        (RED, GREEN, BLUE), THIKNESS)
    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # описанный прямоугольник
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (RED, GREEN, BLUE), THIKNESS)
    return image


if __name__ == '__main__':
    holistic()