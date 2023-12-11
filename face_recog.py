import cv2
import face_recognition
import os

known_face_encodings = []
known_face_names = []

folder_path = "dataset"
known_person_image = [os.path.join(folder_path, file) for file in os.listdir(folder_path) 
               if file.lower().endswith(('.jpg', '.jpeg'))]

load_person_image = []
known_person_encoding = []

for x in range(len(known_person_image)):
    load_person_image.append(face_recognition.load_image_file(known_person_image[x]))
    known_person_encoding.append(face_recognition.face_encodings(load_person_image[x])[0])
    known_face_encodings.append(known_person_encoding[x])

file_name_face = open("name.txt", "r")
known_face_names = file_name_face.read().split(", ")

# buka webcam (0 default)
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()  # capture frame by frame

    # find all face loc in the current frame
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # loop through each face found in the frame
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # check the face its matches or not
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)

        name = "unknown"
        confidence = None

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            confidence = 1 - face_recognition.face_distance([known_face_encodings[first_match_index]], face_encoding)[0]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        text = f"{name}: {confidence:.2f}" if confidence is not None else name
        cv2.putText(frame, text, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # display the result frame
    cv2.imshow("Video", frame)

    # break the loop when q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
video_capture.release()
cv2.destroyAllWindows()
