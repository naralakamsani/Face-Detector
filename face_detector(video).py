import cv2, time

#variable declaration
consistent_face = 0
x_old = 0
y_old = 0
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

while True:
    #Taker the video input
    check, frame = video.read()
    
    #convert to gray scale
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Identify faces from video feed
    faces = face_cascade.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=5)

    for x, y, w, h in faces:
        predicted_group = "50% confident"
        if (x_old + 15 > x > x_old - 15) and (y_old + 15 > y > y_old - 15):
            consistent_face = consistent_face + 1
        else:
            consistent_face = 0

        if consistent_face > 5:
            predicted_group = "100% confident"

        x_old = x
        y_old = y

        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(frame, predicted_group, (x, y - 5), cv2.FONT_ITALIC, 1, (0, 255, 0), 2)

        face_frame = frame[y - 50:y + h + 50, x - 50:x + w + 50]

    cv2.imshow("video", frame)

    # If key is q: exit
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

