import cv2

cap = cv2.VideoCapture(0)
cascade_path = './haarcascades/haarcascade_frontalface_alt2.xml'

while True:
    ret, frame = cap.read()
    if frame is None:
        continue

    cv2.imshow('camera capture', frame)
    cascade = cv2.CascadeClassifier(cascade_path)
    facerect = cascade.detectMultiScale(frame)

    if len(facerect) > 0:
        print(frame)
        print(facerect[0])
        x = facerect[0][0]
        y = facerect[0][1]
        w = facerect[0][2]
        h = facerect[0][3]
        face = frame[y - 40:y + h + 40, x - 30:x + w + 30]
        print(face)
        cv2.imwrite('eval.jpg', face)
        break
    k = cv2.waitKey(10)

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
