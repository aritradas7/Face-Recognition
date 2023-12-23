import face_recognition
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

files = []
imageDir = "IMAGEDB"

for fl in listdir(imageDir):
    if isfile(join(imageDir, fl)):
        if ".jpg" in fl.lower():
            files.append(fl)

known_face_encodings = []
known_face_names = []

for f in files :
    path = join(imageDir, f)
    fimage = face_recognition.load_image_file(path)
    fencode = face_recognition.face_encodings(fimage)[0]
    known_face_encodings.append(fencode)
    known_face_names.append((f.lower().replace(".jpg","")).upper())

video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
name = ""

while True :
    ret, frame = video_capture.read()

    small_frame = cv2.resize(frame, (0, 0), fx=0.75, fy=0.75)

    rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

    if process_this_frame:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
    
    process_this_frame = not process_this_frame
  
    for (top, right, bottom, left), name in zip(face_locations, face_names) :
        cv2.rectangle(small_frame, (left, top), (right, bottom), (0, 255, 0), 1)

        cv2.rectangle(small_frame, (left, bottom - 15), (right, bottom), (0, 255, 0), cv2.FILLED)

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(small_frame, name, (left + 4, bottom - 4), font, 0.35, (0, 0, 0), 1)

    
    grayscale = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
    
    cv2.imshow('Face Recognition Software - Hit "q" to exit!', grayscale)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
