import cv2
import time
import math as m
import mediapipe as mp

def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2-x1)**2 + (y2-y1)**2)
    return dist

def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2-y1)*(-y1) / (m.sqrt(
        (x2-x1)**2 + (y2-y1)**2) * y1
    ))
    degree = int(180/m.pi) *theta
    return degree

# Alerts when bad posture is detected
def sendWarning(x):
    pass

good_frames = 0
bad_frames = 0

# font type
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# file_name = 'input.mp4'
# cap = cv2.VideoCapture(file_name)

cap = cv2.VideoCapture(0)

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = (width, height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

while cap.isOpened():
    success, img = cap.read()

    if not success:
        print('NULL')
        break

    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    keypoints = pose.process(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
    l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)

    r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
    r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

    l_ear_x = int(lm.landmark[lmPose.LEFT_EAR].x * w)
    l_ear_y = int(lm.landmark[lmPose.LEFT_EAR].y * h)

    l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
    l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)

    offset = findDistance(l_shldr_x, l_shldr_y, r_shldr_x, r_shldr_y)

    if offset < 100:
        cv2.putText(img, str(int(offset)) + 'Aligned',
                    (w - 150, 30), font, 0.9, green, 2)
    else:
        cv2.putText(img, str(int(offset)) + 'Not Aligned',
                    (w - 150, 30), font, 0.9, red, 2)

    neck_inclination = findAngle(l_shldr_x, l_shldr_y, l_ear_x, l_ear_y)
    torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)

    cv2.circle(img, (l_shldr_x, l_shldr_y), 7, yellow, -1)
    cv2.circle(img, (l_ear_x, l_ear_y), 7, yellow, -1)
    cv2.circle(img, (l_shldr_x, l_shldr_y - 100), 7, yellow, -1)
    cv2.circle(img, (r_shldr_x, r_shldr_y), 7, pink, -1)
    cv2.circle(img, (l_hip_x, l_hip_y), 7, yellow, -1)

    cv2.circle(img, (l_hip_x, l_hip_y - 100), 7, yellow, -1)

    # Put text, Posture and angle inclination.
    # Text string for display.
    angle_text_string = 'Neck : ' + str(int(neck_inclination)) + '  Torso : ' + str(int(torso_inclination))

    if neck_inclination < 40 and torso_inclination < 10:
        bad_frames = 0
        good_frames += 1

        cv2.putText(img, angle_text_string, (10, 30), font, 0.9, light_green, 2)
        cv2.putText(img, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, light_green, 2)
        cv2.putText(img, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, light_green, 2)

        # Join landmarks.
        cv2.line(img, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), green, 4)
        cv2.line(img, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), green, 4)
        cv2.line(img, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), green, 4)
        cv2.line(img, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), green, 4)

    else:
        good_frames = 0
        bad_frames += 1

        cv2.putText(img, angle_text_string, (10, 30), font, 0.9, red, 2)
        cv2.putText(img, str(int(neck_inclination)), (l_shldr_x + 10, l_shldr_y), font, 0.9, red, 2)
        cv2.putText(img, str(int(torso_inclination)), (l_hip_x + 10, l_hip_y), font, 0.9, red, 2)

        # Join landmarks.
        cv2.line(img, (l_shldr_x, l_shldr_y), (l_ear_x, l_ear_y), red, 4)
        cv2.line(img, (l_shldr_x, l_shldr_y), (l_shldr_x, l_shldr_y - 100), red, 4)
        cv2.line(img, (l_hip_x, l_hip_y), (l_shldr_x, l_shldr_y), red, 4)
        cv2.line(img, (l_hip_x, l_hip_y), (l_hip_x, l_hip_y - 100), red, 4)

        # Calculate the time of remaining in a particular posture.
    good_time = (1 / fps) * good_frames
    bad_time = (1 / fps) * bad_frames

    # Pose time.
    if good_time > 0:
        time_string_good = 'Good Posture Time : ' + str(round(good_time, 1)) + 's'
        cv2.putText(img, time_string_good, (10, h - 20), font, 0.9, green, 2)
    else:
        time_string_bad = 'Bad Posture Time : ' + str(round(bad_time, 1)) + 's'
        cv2.putText(img, time_string_bad, (10, h - 20), font, 0.9, red, 2)

    # If you stay in bad posture for more than 3 minutes (180s) send an alert.
    if bad_time > 180:
        sendWarning()
    # Write frames.
    video_output.write(img)

    # Display.
    cv2.imshow('MediaPipe Pose', img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()