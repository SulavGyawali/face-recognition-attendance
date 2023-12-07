import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
from flask import Flask

app = Flask(__name__)
@app.route("/")
def hello_world():
    return "<p>Hello World</p>"



def main():
    video_capture = cv2.VideoCapture(0)

    #Load known faces
    sulav_image = face_recognition.load_image_file("faces/sulav.jpg")
    sulav_encoding = face_recognition.face_encodings(sulav_image)[0]

    kavi_image = face_recognition.load_image_file("faces/kavi.jpg")
    kavi_encoding = face_recognition.face_encodings(kavi_image)[0]

    suku_image = face_recognition.load_image_file("faces/suku.jpg")
    suku_encoding = face_recognition.face_encodings(suku_image)[0]

    known_face_encoding = [sulav_encoding, kavi_encoding, suku_encoding]
    known_face_names = ["Sulav","Kavi", "Suku`"]

    #List of students
    students = known_face_names.copy()

    face_locations = []
    face_encodings = []

    #Get the current date and time
    now = datetime.now()
    current_date = now.strftime('%Y - %m - %d')

    f = open(f"{current_date}.csv", "w+", newline="")
    lnwriter = csv.writer(f)

    while True:
        _, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0,0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        #Recognize faces
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
            face_distance = face_recognition.face_distance(known_face_encoding, face_encoding)
            best_match_index = np.argmin(face_distance)

        
            if(matches[best_match_index]):
                name = known_face_names[best_match_index]

            #Add the text if the person is present
            if name in known_face_names:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottom_left_text = (10,100)
                font_scale = 1.5
                font_color = (255,0,0)
                thickness = 3
                line_type = 2

                cv2.putText(frame, name + " Present", bottom_left_text,font, font_scale, font_color, thickness, line_type)

                if name in students:
                    current_time = now.strftime("%H - %M - %S")
                    lnwriter.writerow([name, current_time])
                    students.remove(name)

            cv2.imshow("Attendace", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    video_capture.release()
    cv2.destroyAllWindows()
    f.close()

if __name__ == "__main__":
    main()
