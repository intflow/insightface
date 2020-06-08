RTSP_address = "rtsp://admin:kmjeon3121@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0"

import rtsp
import cv2
import numpy as np
import imutils

with rtsp.Client(rtsp_server_uri=RTSP_address) as client:
    while True:
        _image = client.read(raw=True)
        # print(type(_image))
        # print(_image)

        if _image is not None:
            _image = imutils.resize(_image, width=460)
            cv2.imshow('frame', _image)
            # cv2.imshow('rtsp', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    client.close()
cv2.destroyAllWindows()





