#imports
import cv2, time

#Use CascadeClassifier  to define how a face looks like to the computer
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Start capturing Video
video = cv2.VideoCapture(0)

while True:
    check, frame = video.read()
    
    #Apply grey filter to each frame
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #Detect if the current frame contains a face
    faces = face_cascade.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=5)
    
    #Box the face found in the frame
    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    #Show the frame after box on the face has been applied
    cv2.imshow("video", frame)


    key = cv2.waitKey(1)

    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
