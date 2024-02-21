import cv2
import face_recognition
import numpy as np
import os
import json

known_face_encodings = []
known_face_names = []

dataset_folder = 'dataset'
image_files = os.listdir(dataset_folder)

for image_file in image_files:
    # Load gambar dan encoding wajah
    image_path = os.path.join(dataset_folder, image_file)
    person_image = face_recognition.load_image_file(image_path)
    person_encoding = face_recognition.face_encodings(person_image)[0]

    # Tambahkan encoding dan nama ke list
    known_face_encodings.append(person_encoding)
    known_face_names.append(image_file.split('.')[0])

encodings_str = [str(encoding) for encoding in known_face_encodings]

data_str = [name + "=" + encoding for name, encoding in zip(known_face_names, encodings_str)]
People_Face_Encodings = r"Face_encodings.json"

# Open the JSON file and write data to it
with open(People_Face_Encodings, 'w+') as json_file:
    json.dump(data_str, json_file, indent=4)

# buka webcam (0 default)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()  # capture frame by frame

    # find all face loc in the current frame
    face_locations = face_recognition.face_locations(frame)

    # find the bigger face
    bigger_face, bigger_size = -1, -1
    for number, (top, right, bottom, left) in enumerate(face_locations):
        if((bottom - top) * (right - left)) >= bigger_size:
            bigger_size = (bottom - top) * (right - left)
            bigger_face = number

    if bigger_face == -1: # No face detected
        pass
    else:
        face_locations = [face_locations[bigger_face]]

        top, right, bottom, left = face_locations[0]
        face_encoding = face_recognition.face_encodings(frame, face_locations)[0]

        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        name = "unknown"

        index = np.argmin(distances)

        if True in matches:
            if matches[index]:
                first_match_index = matches.index(True)
                name = known_face_names[index]

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # display the result frame
    cv2.imshow("Video", frame)

    # break the loop when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release cam and close opencv windws
video_capture.release()
cv2.destroyAllWindows