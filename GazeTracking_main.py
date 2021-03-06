import cv2
import numpy as np
import dlib
import time
from math import hypot
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

font = cv2.FONT_HERSHEY_PLAIN

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def get_blinking_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))

    #hor_line = cv2.line(frame, left_point, right_point, (0, 255, 0), 2)
    #ver_line = cv2.line(frame, center_top, center_bottom, (0, 255, 0), 2)

    hor_line_lenght = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    ver_line_lenght = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))

    if ver_line_lenght != 0:
        ratio = hor_line_lenght / ver_line_lenght
        return ratio







def get_gaze_ratio(eye_points, facial_landmarks):
    left_eye_region = np.array([(facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y),
                                (facial_landmarks.part(eye_points[1]).x, facial_landmarks.part(eye_points[1]).y),
                                (facial_landmarks.part(eye_points[2]).x, facial_landmarks.part(eye_points[2]).y),
                                (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y),
                                (facial_landmarks.part(eye_points[4]).x, facial_landmarks.part(eye_points[4]).y),
                                (facial_landmarks.part(eye_points[5]).x, facial_landmarks.part(eye_points[5]).y)], np.int32)
    cv2.polylines(frame, [left_eye_region], True, (0, 0, 255), 2)

    height, width, _ = frame.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.polylines(mask, [left_eye_region], True, 255, 2)
    cv2.fillPoly(mask, [left_eye_region], 255)
    eye = cv2.bitwise_and(gray, gray, mask=mask)

    min_x = np.min(left_eye_region[:, 0])
    max_x = np.max(left_eye_region[:, 0])
    min_y = np.min(left_eye_region[:, 1])
    max_y = np.max(left_eye_region[:, 1])

    gray_eye = eye[min_y:max_y, min_x:max_x]
    _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
    height, width = threshold_eye.shape
    left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
    left_side_white = cv2.countNonZero(left_side_threshold)

    right_side_threshold = threshold_eye[0: height, int(width / 2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    if left_side_white == 0:
        gaze_ratio = 1
    elif right_side_white == 0:
        gaze_ratio = 5
    else:
        gaze_ratio = left_side_white / right_side_white

    return gaze_ratio



A = np.zeros(29, np.uint8)

while True:
    _, frame = cap.read()
    new_frame = np.zeros((500,500,3) , np.uint8)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    for face in faces:
        landmarks = predictor(gray, face)


        left_eye_ratio = get_blinking_ratio([36,37,38,39,40,41], landmarks)
        right_eye_ratio = get_blinking_ratio([42,43,44,45,46,47], landmarks)
        blinking_ratio =int( (left_eye_ratio+right_eye_ratio)/2 )
        if blinking_ratio > 5.7 :
            cv2.putText(frame, "Blinking", (450,150), font, 3,(255,0,0))

        gaze_ratio_left_eye = get_gaze_ratio([36, 37, 38, 39, 40, 41], landmarks)
        gaze_ratio_right_eye = get_gaze_ratio([42, 43, 44, 45, 46, 47], landmarks)
        gaze_ratio = (gaze_ratio_right_eye + gaze_ratio_left_eye) / 2


        #print(gaze_ratio)
        if(gaze_ratio>2):
            gaze_ratio=2
        if(gaze_ratio<0.5):
            gaze_ratio=0.5

        clr = 50 + 1350*((gaze_ratio-0.5)/1.5)
        #print(clr)
        new_frame = cv2.resize(new_frame,(1400,400))
        l = clr-25
        r = clr+25
        new_frame[:,int(l):int(r)] = (0,0,255)

        clr = int(clr/50)
        #print(clr)
        A[int(clr)] = int(A[int(clr)]+1)





    #time.sleep(1)

    cv2.imshow("GazeDistribution on the monitor",new_frame)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break



print(A)

# x-coordinates of left sides of bars
left = [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28]

# heights of bars
A

# labels for bars

# plotting a bar chart
plt.bar(left, A,
        width=0.8, color=['red'])

# naming the x-axis
plt.xlabel('x - axis')
# naming the y-axis
plt.ylabel('y - axis')
# plot title
plt.title('Gaze Distribution!')

# function to show the plot
plt.show()