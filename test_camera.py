import cv2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
for i in range(10):
    ret, frame = cap.read()
    print(i, ret, frame.mean() if ret else None)
cv2.imwrite("test_frame.jpg", frame)
cap.release()