import cv2
from deepface import DeepFace


faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    raise IOError("Webcam error")


while True:
    ret,frame = cam.read()
    result = DeepFace.analyze(frame, actions = ["emotion", "gender", "race"])

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.1, 4)


    for(x, y, w, h ) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + x), (0, 255, 0), 2)

    font = cv2.FONT_HERSHEY_DUPLEX


    cv2.putText(frame,
               result['gender'],
               (100, 50),
               font, 1,
               (220, 0, 0),
               1,
               cv2.LINE_4)
    cv2.putText(frame,
               result['dominant_race'],
               (100, 100),
               font, 1,
               (220, 0, 0),
               1,
               cv2.LINE_4)
    cv2.putText(frame,
               result['dominant_emotion'],
               (100, 150),
               font, 1,
               (220, 0, 0),
               1,
               cv2.LINE_4)

    
                
    cv2.imshow("Demo Video", frame)

    if  cv2.waitKey(2) & 0xFF == ord("e"):
        break


cam.release()
cv2.destroyAllWindows()
