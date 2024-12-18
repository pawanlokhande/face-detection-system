import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = r'F:\face_reconition_system\docu\final project\images'
images = []
classNames = []
myList = os.listdir(path)

# Load images and corresponding names
for cl in myList:
    curImg = cv2.imread(os.path.join(path, cl))
    if curImg is not None:
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    else:
        print(f"Image {cl} is empty or invalid.")

# Create or append to CSV for attendance
with open('$$$$$$$$$$.csv', 'w') as f:
    f.write('Name,Time\n')

# Function to find face encodings
def findEncodings(images):
    encodeList = []
    for i, img in enumerate(images):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encode = face_encodings[0]
            encodeList.append(encode)
        else:
            print(f"Skipping image {i + 1} due to no face detection.")
    return encodeList

# Function to mark attendance
def markAttendance(name):
    with open('21.12.2023.csv', 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%H:%M:%S')
        f.write(f'{name},{dtString}\n')

# Get known encodings
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Start webcam capture
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from the webcam")
        break

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
        else:
            name = "unknown person"

        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

        # Draw rectangle and display name on webcam
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

        # Mark attendance for recognized face
        markAttendance(name)

    # Display the image
    cv2.imshow('Webcam', img)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close windows
cap.release()
cv2.destroyAllWindows()
