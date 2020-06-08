import cv2
import imutils
from imutils.video import VideoStream
# cap = cv2.VideoCapture('rtsp://admin:kmjeon3121@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0')
cap = VideoStream('rtsp://admin:kmjeon3121@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0').start()

# if cap.isOpened():
#     print('width: {}, height : {}'.format(cap.get(3), cap.get(4)))

while True:
    frame = cap.read()
    # ret, frame = cap.read()
    
    if frame is not None:
        frame = imutils.resize(frame, width=460)
        cv2.imshow('rtsp', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    else:
        print('error')

cap.release()
cv2.destroyAllWindows()
