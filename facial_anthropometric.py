import cv2
import mediapipe as mp
import numpy as np
from google.colab.patches import cv2_imshow

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Image dimensions
TARGET_WIDTH = 463
TARGET_HEIGHT = 550

# Load your image
image_path = '/content/Picture1.jpg'  # Replace with your image path
image = cv2.imread(image_path)

# Resize the image to the target size
resized_image = cv2.resize(image, (TARGET_WIDTH, TARGET_HEIGHT))

# Convert the image to RGB as MediaPipe works with RGB format
rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

# Perform face landmark detection
results = face_mesh.process(rgb_image)

# Get the face landmarks (if detected)
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Get the coordinates of Point 127 and Point 356
        h, w, _ = resized_image.shape  # Get the resized image dimensions

        # Point 127 - Left side of the forehead
        point_54 = face_landmarks.landmark[54]
        x1, y1 = int(point_54.x * w), int(point_54.y * h)
        # Point 356 - Right side of the forehead
        point_284= face_landmarks.landmark[284]
        x2, y2 = int(point_284.x * w), int(point_284.y * h)
        # Calculate the Manhattan distance
        forehead_manhattan_distance = abs(x2 - x1) + abs(y2 - y1)

        point_149 = face_landmarks.landmark[136]
        x3, y3 = int(point_149.x * w), int(point_149.y * h)
        # Point 378 - Right side of the chin
        point_378 = face_landmarks.landmark[365]
        x4, y4 = int(point_378.x * w), int(point_378.y * h)
        # Calculate the Manhattan distance for the chin width
        chin_manhattan_distance = abs(x4 - x3) + abs(y4 - y3)

        point_168 = face_landmarks.landmark[8]
        x5,y5 = int(point_168.x*w),int(point_168.y*h)
        point_152 = face_landmarks.landmark[175]
        x6,y6= int(point_152.x*w), int(point_152.y*h)
        MNBL_manhattan_distance = abs(x6 - x5) + abs(y6 - y5)

        point_168 = face_landmarks.landmark[168]
        x7,y7 = int(point_168.x*w),int(point_168.y*h)
        point_152 = face_landmarks.landmark[152]
        x8,y8= int(point_152.x*w), int(point_152.y*h)
        face_manhattan_distance = abs(x8 - x7) + abs(y8 - y7)

        point_168 = face_landmarks.landmark[1]
        x9,y9 = int(point_168.x*w),int(point_168.y*h)
        point_152 = face_landmarks.landmark[152]
        x10,y10= int(point_152.x*w), int(point_152.y*h)
        Lowerface_manhattan_distance = abs(x10 - x9) + abs(y10 - y9)

        point_168 = face_landmarks.landmark[0]
        x11,y11 = int(point_168.x*w),int(point_168.y*h)
        point_152 = face_landmarks.landmark[18]
        x12,y12= int(point_152.x*w), int(point_152.y*h)
        LipWidth_manhattan_distance = abs(x12 - x11) + abs(y12 - y11)

        point_168 = face_landmarks.landmark[18]
        x13,y13 = int(point_168.x*w),int(point_168.y*h)
        point_152 = face_landmarks.landmark[175]
        x14,y14= int(point_152.x*w), int(point_152.y*h)
        MCL_manhattan_distance = abs(x14 - x13) + abs(y14 - y13)

        print(f"Manhattan Distance: ")
        print(f"head width: {forehead_manhattan_distance}")
        print(f"chin width: {chin_manhattan_distance}")
        print(f"Menton-Nasal bridge length: {MNBL_manhattan_distance}")
        print(f"face length: {face_manhattan_distance}")
        print(f"Lower face length: {Lowerface_manhattan_distance}")
        print(f"Lip Width: {LipWidth_manhattan_distance}")
        print(f"Menton Chin Length: {MCL_manhattan_distance}")

        # Optionally, display the points on the image
        cv2.circle(resized_image, (x1, y1), 5, (0, 255, 0), -1)  # Green for Point 127
        cv2.circle(resized_image, (x2, y2), 5, (0, 0, 255), -1)  # Red for Point 356

        cv2.circle(resized_image, (x3, y3), 5, (255, 0, 0), -1)  # Blue for Point 149 (chin)
        cv2.circle(resized_image, (x4, y4), 5, (255, 255, 0), -1)

        cv2.circle(resized_image, (x5, y5), 5, (255, 255, 0), -1)  # Blue for Point 149 (chin)
        cv2.circle(resized_image, (x6, y6), 5, (160, 32, 240), -1)

        cv2.circle(resized_image, (x7, y7), 5, (2, 0, 0), -1)  # Blue for Point 149 (chin)
        cv2.circle(resized_image, (x8, y8), 5, (255, 255, 0), -1)

        cv2.circle(resized_image, (x9, y9), 5, (255,165, 0), -1)  # Blue for Point 149 (chin)
        cv2.circle(resized_image, (x10, y10), 5, (255,141,161), -1)

        cv2.circle(resized_image, (x11, y11), 5, (0, 255, 0), -1)  # Green for Point 127
        cv2.circle(resized_image, (x12, y12), 5, (0, 0, 255), -1)  # Red for Point 356

        cv2.circle(resized_image, (x13, y13), 5, (255, 0, 0), -1)  # Blue for Point 149 (chin)
        cv2.circle(resized_image, (x14, y14), 5, (255, 255, 0), -1)

# Show the resulting image with points marked
cv2_imshow(resized_image)