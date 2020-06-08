"""
*Initial writer : GBKim
*Date           : Mon.25.May.2020
*Description    :
    - It contains the code for Mask_face detection trough rtsp.
    - You need to check parsed arguments.
*logs           :
    - GBKim 05.25.2020: Initial coding.
"""

import cv2
import sys
import numpy as np
import datetime
import os
import glob
import time
import argparse
import imutils
from imutils.video import VideoStream
from imutils.video import WebcamVideoStream
from imutils.video import FPS
from retinaface_cov import RetinaFaceCoV

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    ## Parse arguments
    parser = argparse.ArgumentParser(description="argparser for mask_face_detection.")

    parser.add_argument("-S", "--source", default="rtsp://admin:kmjeon3121@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0", type=str, help="Path to the rtsp address of webcam or the path of the video.")
    parser.add_argument("-MP", "--model_path", default="./model/mnet_cov2")
    parser.add_argument("-TH", "--threshold", default=0.8, type=float, help="")
    parser.add_argument("-MTH", "--mask_thresh", default=0.2, type=float, help="")
    parser.add_argument("--epoch", default=0, type=int, help="model's epoch.")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID.")
    parser.add_argument("--frame_width", default=480, type=int, help="resize width.")
    parser.add_argument("--frame_height", default=270, type=int, help="resize width.")
    parser.add_argument("--frame_flip", default=False, type=str2bool, help="Flip frame or not.")
    args = vars(parser.parse_args())

    ## Default variables
    scales = [270, 480] # [640, 960]
    target_size = scales[0] #640
    max_size = scales[1] #1080
    count = 2

    ## Load model 
    detector = RetinaFaceCoV(prefix=args['model_path'], epoch=args['epoch'], ctx_id=args['gpu_id'], network='net3l')

    ## Capture RTSP video
    vs = VideoStream(src=args['source']).start()
    frame = vs.read()
    frame = imutils.resize(frame, width=args["frame_width"], height=args["frame_height"])
    print(f"[INFO] frame size: ({frame.shape[0]},{frame.shape[1]})")

    ## Setting for frame resize
    im_size_min = np.min(frame.shape[0:2])
    im_size_max = np.max(frame.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size: # prevent bigger axis from being more than max_size
        im_scale = float(max_size) / float(im_size_max)
    print(f"[INFO] image scale: {im_scale}")
    scales = [im_scale]
    flip = args['frame_flip']

    prevTime = 0 # For FPS

    ## Display Frame
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=args["frame_width"], height=args["frame_height"])

        curTime = time.time() # For FPS

        for c in range(count):
            faces, landmarks = detector.detect(frame, args["threshold"], scales=scales, do_flip=flip)

        if faces is not None:
            print(f"Find, {faces.shape[0]} faces.")

            for i in range(faces.shape[0]):
                face = faces[i]
                box = face[0:4].astype(np.int)
                mask = face[5]
                print(i, box, mask)

                if mask >= args["mask_thresh"]:
                    color = (0,0,255)
                else:
                    color = (0,255,0)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 5)
                landmark5 = landmarks[i].astype(np.int)

                for l in range(landmark5.shape[0]):
                    color = (255,0,0)
                    cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), 1, color, 5)

        # check frame
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / sec
        fps_str_format = "FPS: {%.2f}" % fps 
        print("[INFO] approx. current FPS: {:.2f}".format(fps))
        cv2.putText(frame, fps_str_format, (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        dispaly_frame = imutils.resize(frame.copy(), width=480, height=270)
        cv2.imshow("RTSP frame", dispaly_frame)



        # Press 'q' to terminate cv2.imshow()
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()

        

        






