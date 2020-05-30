"""
*Initial writer : GBKim
*Date           : Mon.26.May.2020
*Description    :
    - It contains the code for Face Feature (5points) extractor and Face   
        detector on rtsp frame.
    - You need to check parsed arguments.
*logs           :
    - GBKim 05.25.2020: Initial coding.
"""
## Import Packages
import cv2
import sys
import argparse
import imutils
import time
import numpy as np
from imutils.video import VideoStream
from imutils.video import FPS
from retinaface import RetinaFace

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    ## Default variables
    scales = [270, 480] #[1024, 1980]
    count = 1
    target_size = scales[0]
    max_size = scales[1]

    ## Load model 
    detector = RetinaFace(prefix=args['model_path'], epoch=args['epoch'], ctx_id=args['gpu_id'], network='net3')

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
            face_bbox, landmarks = detector.detect(frame, args["threshold"], scales=scales, do_flip=flip)

        if face_bbox is not None:
            print(f"Find, {face_bbox.shape[0]} face_bbox.")

            for i in range(face_bbox.shape[0]):
                box = face_bbox[i].astype(np.int)
                print(f"[INFO] Information {i+1}-th face box: {box}")
                color = (0,0,255)

                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 5)

                if landmarks is not None:
                    landmark5 = landmarks[i].astype(np.int)
                    print(f"[INFO] Information {i+1}-th face landmarks: {landmark5}")

                for l in range(landmark5.shape[0]):
                    color = (255,0,0)
                    cv2.circle(frame, (landmark5[l][0], landmark5[l][1]), args["circle_size"], color, 5)

        # check frame
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / sec
        fps_str_format = "FPS: {%.2f}" % fps 
        print("[INFO] approx. current FPS: {:.2f}".format(fps))
        cv2.putText(frame, fps_str_format, (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

        dispaly_frame = imutils.resize(frame.copy(), width=480, height=270)
        cv2.imshow("RTSP frame", dispaly_frame)
        
        # Press 'q' to terminate cv2.imshow()
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    ## Parse arguments
    parser = argparse.ArgumentParser(description="argparser for mask_face_detection.")

    parser.add_argument("-S", "--source", default="rtsp://admin:kmjeon3121@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0", type=str, help="Path to the rtsp address of webcam or the path of the video.")
    parser.add_argument("-MP", "--model_path", default="/home/gbkim/gb_dev/insightface_MXNet/insightface/RetinaFace/model/retinaface-R50/R50", type=str, help="Path of the model.")
    parser.add_argument("-TH", "--threshold", default=0.8, type=float, help="detection threshold")
    parser.add_argument("--epoch", default=0, type=int, help="model's epoch.")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID.")
    parser.add_argument("--frame_width", default=480, type=int, help="resize width.")
    parser.add_argument("--frame_height", default=270, type=int, help="resize width.")
    parser.add_argument("--frame_flip", default=False, type=str2bool, help="Flip frame or not.")
    parser.add_argument("--circle_size", default=2, type=int, help="size of circle radius for landmarks on the face")
    args = vars(parser.parse_args())

    main(args)

