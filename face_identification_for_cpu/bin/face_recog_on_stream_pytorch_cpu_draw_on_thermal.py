import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'architecture'))

import cv2
import time
import dlib
import rtsp
import torch
import torchsummary
import pickle
import sklearn
import imutils
import argparse
import numpy as np
import mxnet as mx
import matplotlib.pyplot as plt
from utils import face_preprocess
from imutils.video import VideoStream
from architecture.retinaface import RetinaFace
from utils.face_detection_and_align import AnalyzeFace
from architecture.embedding_learner_pytorch_cpu import embedding_classifier


def str2bool(v):
    """convert string to bool for argparser

    Args:
        v (str)

    Raises:
        argparse.ArgumentTypeError: [description]

    Returns:
        boolean
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def findCosineDistance(vector1, vector2):
    """
    Calculate cosine distance between two vector
    """
    vec1 = vector1.flatten()
    vec2 = vector2.flatten()

    a = np.dot(vec1.T, vec2)
    b = np.dot(vec1.T, vec1)
    c = np.dot(vec2.T, vec2)
    return 1 - (a/(np.sqrt(b)*np.sqrt(c)))

def CosineSimilarity(test_vec, source_vecs):
    """
    Verify the similarity of one vector to group vectors of one class
    """
    cos_dist = 0
    for source_vec in source_vecs:
        cos_dist += findCosineDistance(test_vec, source_vec)
    return cos_dist/len(source_vecs)

def load_embedding_model(model_str, epoch, ctx, image_size_for_align, layer):
    """load mxnet model

    Args:
        model_str (str): prefix of mxnet model
        epoch (int): mode's epoch
        ctx (gpu id): allocate device
        image_size (list or tuple): (width, height)
        layer (str): name of layer

    Returns:
        model (mxnet model)
    """

    _vec_image_size = image_size_for_align.split(',')
    assert len(_vec_image_size) == 2
    image_size = (int(_vec_image_size[0]), int(_vec_image_size[1]))

    # image_size = (int(image_height), int(image_width))

    prefix = model_str
    epoch = int(epoch)
    
    sym, arg_params, aux_params = mx.model.load_checkpoint(model_str, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']

    if isinstance(ctx, int) and ctx>=0:
        model = mx.mod.Module(symbol=sym, context=mx.gpu(ctx), label_names=None)
    else:
        model = mx.mod.Module(symbol=sym, context=mx.cpu(), label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    return model

def get_feature(embedding_model, aligned_img):
    input_img = np.expand_dims(aligned_img, axis=0)
    data = mx.nd.array(input_img)
    db = mx.io.DataBatch(data=(data,))
    embedding_model.forward(db, is_train=False)
    embedding = embedding_model.get_outputs()[0].asnumpy()
    embedding = sklearn.preprocessing.normalize(embedding).flatten()
    
    return embedding

def img_ratio_converter(x, y, r_x, r_y):
    x_hat = np.abs(int(x * r_x) - 1)
    y_hat = np.abs(int(y * r_y) - 1)
    
    return x_hat, y_hat


def main(args, scale_candidate):
    """[summary]

    Args:
        args ([type]): [description]
        scale_candidate ([type]): [description]
    Return:
        cv2.imshow()
    """
    ## Default arguments
    use_cuda = torch.cuda.is_available()
    gpu_device = torch.device('cuda')
    cpu_device = torch.device('cpu')
    count = 2
    target_size = scale_candidate[0]
    max_size = scale_candidate[1]

    ## Load embeddings and labels
    embedding_data = pickle.loads(open(args["embeddings_path"], "rb").read())
    label_data = pickle.loads(open(args["encoded_label_path"], "rb").read())

    ## Set data
    input_embeddings = np.array(embedding_data["embeddings"])
    labels = label_data.fit_transform(embedding_data['names'])
    num_classes = len(np.unique(labels))

    ## Load face detector
    face_detector = RetinaFace(prefix=args['face_detect_model_path'], epoch=args['detector_epoch'], ctx_id=args['gpu_id'], network='net3')
    print("[INFO] RetinaFace has been loaded as a face_detector...")

    ## Load embedding extractor on cpu
    embedding_model = load_embedding_model(model_str=args['embedding_model_path'], epoch=args['embedding_epoch'], ctx=args['gpu_id'], image_size_for_align=args['image_size_for_align'], layer='fc1')
    print("[INFO] embedding model has been loaded as a embedding extractor...")

    ## Load the pytorch classifier model on cpu
    embedding_classifier_model = embedding_classifier(input_embeddings.shape[1], num_classes=num_classes)
    embedding_classifier_model.load_state_dict(torch.load(args['classifier_path'],  map_location=cpu_device))
    embedding_classifier_model.eval()
    torchsummary.summary(embedding_classifier_model, (input_embeddings.shape[1], ))## show summary of model
    print("[INFO] pytorch embedding classifier model has been loaded...")

    ## Capture video streaming
    normal_vs = rtsp.Client(rtsp_server_uri=args['source1'])
    thermal_vs = rtsp.Client(rtsp_server_uri=args['source2'])
    while True:
        normal_frame = normal_vs.read(raw=True)
        thermal_frame = thermal_vs.read(raw=True)
        if (normal_frame is not None) & (thermal_frame is not None):
            break

    ## Resize frames
    normal_frame = imutils.resize(normal_frame, width=args["resize_width"])
    thermal_frame = imutils.resize(thermal_frame, height=args['resize_height'])

    ## Get Center point of thermal image
    thermal_img_Cx = int(thermal_frame.shape[1]/2)
    thermal_img_Cy = int(thermal_frame.shape[0]/2)

    ## get image difference ratio for bounding box
    r_x = thermal_frame.shape[1] / normal_frame.shape[1]
    r_y = thermal_frame.shape[0] / normal_frame.shape[0]

    print(f"[INFO] normal frame size: ({normal_frame.shape[0]},{normal_frame.shape[1]})")
    print(f"[INFO] normal frame size: ({thermal_frame.shape[0]},{thermal_frame.shape[1]})")
    

    ## Initialize arguments
    comparing_num = 5
    trackers = []
    texts = []
    frames = 0 #count frame
    prevTime = 0 # for FPS

    ## Setting for frame resize to detect face
    im_size_min = np.min(normal_frame.shape[0:2])
    im_size_max = np.max(normal_frame.shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size: # prevent bigger axis from being more than max_size
        im_scale = float(max_size) / float(im_size_max)
    print(f"[INFO] image scale: {im_scale}")
    scales = [im_scale]

    ## main loop
    while True:
        ## Read frames from python rtsp class
        normal_frame = normal_vs.read(raw=True)
        thermal_frame = thermal_vs.read(raw=True)

        frames += 1 #count current frame number

        ## Resize
        normal_frame = imutils.resize(normal_frame, width=args["resize_width"])
        thermal_frame = imutils.resize(thermal_frame, height=args['resize_height'])
        print('[INFO] Current size of Normal Frame: ', normal_frame.shape)
        print('[INFO] Current size of Thermal Frame: ', thermal_frame.shape)


        currTime = time.time()

        ## Do detection and set tracker.
        if int(frames % args['frame_num_for_detection']) == 0:
            trackers = []
            texts = []

            ## detect faces
            face_bbox, landmarks = face_detector.detect(normal_frame, args["det_threshold"], scales=scales, do_flip=args['frame_flip'])
            
            if not int(face_bbox.shape[0]) == 0:
                print(f"Find, {face_bbox.shape[0]} face_bbox.")

                for i in range(face_bbox.shape[0]):
                    box = face_bbox[i].astype(np.int)
                    print(f"[INFO] Information {i+1}-th face box: {box}")

                    each_landmark = landmarks[i].astype(np.int)

                    ## image warping
                    warped_img = face_preprocess.preprocess(normal_frame, bbox=box, landmark=each_landmark, image_size=args['image_size_for_align'])
                    warped_img = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)
                    warped_img = np.transpose(warped_img, (2,0,1))
                    
                    ## extract embedings
                    extrated_embedding = get_feature(embedding_model=embedding_model, aligned_img=warped_img)
                    extrated_embedding = extrated_embedding.reshape(1, -1)

                    ## Predcit class
                    preds = embedding_classifier_model(extrated_embedding)
                    preds = preds.flatten()

                    ## Get the highest accuracy embedded vector
                    candidate_index = torch.argmax(preds).item()
                    candidate = preds[candidate_index]

                    ## Compare this vector to source class vectors to verify it is actual belong to this class
                    match_class_idx = (labels == candidate_index)
                    match_class_idx = np.where(match_class_idx)[0]
                    selected_idx = np.random.choice(match_class_idx, comparing_num)
                    compare_embeddings = input_embeddings[selected_idx]
                    
                    ## Calculate cosine similarity
                    cos_similarity = CosineSimilarity(extrated_embedding, compare_embeddings)

                    text = 'Unknown'

                    if cos_similarity < args['cosine_threshold'] and candidate > args['proba_threshold']:
                        name = label_data.classes_[candidate_index]
                        text = "{}".format(name)
                        print("Recognized: {} <{:.2f}>".format(name, candidate*100))

                    ## Start tracking
                    tracker = dlib.correlation_tracker()
                    rect = dlib.rectangle(box[0], box[1], box[2], box[3])
                    # tracker.start_track(frame_copy, rect)
                    tracker.start_track(normal_frame, rect)
                    trackers.append(tracker)
                    texts.append(text)

                    ## Display box and texts
                    i = 0
                    for tracker, text in zip(trackers, texts):
                        i += 1

                        pos = tracker.get_position()

                        ## bounding box for normal frame
                        startX = int(pos.left())
                        startY = int(pos.top())
                        endX = int(pos.right())
                        endY = int(pos.bottom())
                        
                        ## Resize bounding box for thermal frame
                        thermal_startX, thermal_startY, thermal_endX, thermal_endY = int(startX*r_x - 9), int(startY*r_y - 5), int(endX*r_x + 9), int(endY*r_y + 5)

                        ## set position of name
                        name_txt_y_pose = startY - 10 if startY - 10 > 10 else startY + 10
                        thermal_txt_y_pose = thermal_startY - 10 if thermal_startY - 10 > 10 else thermal_startY + 10

                        ## Get face pixels from thermal frame inside of the bbox
                        thermal_img_inside_bbox = thermal_frame[thermal_startY:thermal_endY][:, thermal_startX:thermal_endX][:,:,:].copy()

                        if thermal_img_inside_bbox.size !=0:

                            ## Get center of thermal face image to move box
                            thermal_face_Cx = int((thermal_endX + thermal_startX)/2)
                            thermal_face_Cy = int((thermal_endY + thermal_startY)/2)

                            if (thermal_img_Cx - thermal_face_Cx) > 0:
                                ## move box
                                diff_img_center_to_box_center = int((thermal_img_Cx - thermal_face_Cx)/3)
                                thermal_startX = thermal_startX - diff_img_center_to_box_center
                                thermal_endX = thermal_endX - diff_img_center_to_box_center
                                ## Get center of thermal face image to move box
                                thermal_face_Cx = int((thermal_endX + thermal_startX)/2)
                                thermal_face_Cy = int((thermal_endY + thermal_startY)/2)
                            elif (thermal_img_Cx - thermal_face_Cx) < 0:
                                ## move box
                                diff_img_center_to_box_center = int((thermal_img_Cx - thermal_face_Cx)/2)
                                thermal_startX = thermal_startX - diff_img_center_to_box_center
                                thermal_endX = thermal_endX - diff_img_center_to_box_center
                                ## Get center of thermal face image to move box
                                thermal_face_Cx = int((thermal_endX + thermal_startX)/2)
                                thermal_face_Cy = int((thermal_endY + thermal_startY)/2)

                            ## Get face pixels from thermal frame inside of the bbox
                            thermal_img_inside_bbox = thermal_frame[thermal_startY+10:thermal_endY-10][:, thermal_startX+10:thermal_endX-10][:,:,:].copy()
                           
                            ## Get max position inside of bbox using R-channel and G-channel 
                            rescaled_thermal_face_area = thermal_img_inside_bbox[:][:,:][:,:,2].astype(np.uint16)*10 + thermal_img_inside_bbox[:][:,:][:,:,1].astype(np.uint16)

                            if rescaled_thermal_face_area.size != 0:
                                thermal_max_index = np.where(rescaled_thermal_face_area == rescaled_thermal_face_area.max())
                                try:
                                    for x, y in zip(thermal_max_index[0], thermal_max_index[1]):
                                        cv2.drawMarker(thermal_frame, (x + thermal_startX+10, y + thermal_startY+10), (0, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=5, thickness=1)
                                except:
                                    max_coordinate = (int(thermal_max_index[0][0]), int(thermal_max_index[1][0]))

                                    cv2.drawMarker(thermal_frame, (max_coordinate[0] + thermal_startX + 10, max_coordinate[1] + thermal_startY + 10), (0, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=5, thickness=1)
                                    
                            ## draw vertical line on thermal image
                            cv2.line(thermal_frame, (thermal_img_Cx, 0), (thermal_img_Cx, thermal_frame.shape[0]) , (0, 255, 0), 2)

                            ## drawing center of bounding box
                            cv2.circle(thermal_frame, (thermal_face_Cx, thermal_face_Cy), 2, (0, 255, 0), -1)

                        ## draw bounding box
                        cv2.rectangle(normal_frame, (startX, startY), (endX, endY), (0,255,0), 2)
                        cv2.rectangle(thermal_frame, (thermal_startX, thermal_startY), (thermal_endX, thermal_endY), (0,255,0), 2)

                        ## draw name on each frame
                        cv2.putText(normal_frame, text, (startX, name_txt_y_pose), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(thermal_frame, text, (thermal_startX, thermal_txt_y_pose), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        ## Use tracker
        elif len(trackers) !=0 and len(texts) !=0 :
            
            for tracker, text in zip(trackers, texts):
                pos = tracker.get_position()

                ## bounding box for normal frame
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                ## bounding box for thermal frame
                thermal_startX, thermal_startY, thermal_endX, thermal_endY = int(startX*r_x - 9), int(startY*r_y - 5), int(endX*r_x + 9), int(endY*r_y + 5)

                ## set position of name
                name_txt_y_pose = startY - 10 if startY - 10 > 10 else startY + 10
                thermal_txt_y_pose = thermal_startY - 10 if thermal_startY - 10 > 10 else thermal_startY + 10

                ## Get face pixels from thermal frame inside of the bbox
                thermal_img_inside_bbox = thermal_frame[thermal_startY:thermal_endY][:, thermal_startX:thermal_endX][:,:,:].copy()

                if thermal_img_inside_bbox.size !=0:

                    thermal_face_Cx = int((thermal_endX + thermal_startX)/2)
                    thermal_face_Cy = int((thermal_endY + thermal_startY)/2)

                    if (thermal_img_Cx - thermal_face_Cx) > 0:
                        diff_img_center_to_box_center = int((thermal_img_Cx - thermal_face_Cx)/3)
                        thermal_startX = thermal_startX - diff_img_center_to_box_center
                        thermal_endX = thermal_endX - diff_img_center_to_box_center
                        ## Get center of thermal face image to move box
                        thermal_face_Cx = int((thermal_endX + thermal_startX)/2)
                        thermal_face_Cy = int((thermal_endY + thermal_startY)/2)
                    elif (thermal_img_Cx - thermal_face_Cx) < 0:
                        diff_img_center_to_box_center = int((thermal_img_Cx - thermal_face_Cx)/2)
                        thermal_startX = thermal_startX - diff_img_center_to_box_center
                        thermal_endX = thermal_endX - diff_img_center_to_box_center
                        ## Get center of thermal face image to move box
                        thermal_face_Cx = int((thermal_endX + thermal_startX)/2)
                        thermal_face_Cy = int((thermal_endY + thermal_startY)/2)

                    ## Get face pixels from thermal frame inside of the bbox
                    thermal_img_inside_bbox = thermal_frame[thermal_startY+10:thermal_endY-10][:, thermal_startX+10:thermal_endX-10][:,:,:].copy()

                    ## Get max position inside of bbox
                    rescaled_thermal_face_area = thermal_img_inside_bbox[:][:,:][:,:,2].astype(np.uint16)*10 + thermal_img_inside_bbox[:][:,:][:,:,1].astype(np.uint16)

                    if rescaled_thermal_face_area.size != 0:
                        thermal_max_index = np.where(rescaled_thermal_face_area == rescaled_thermal_face_area.max())

                        try:
                            for x, y in zip(thermal_max_index[0], thermal_max_index[1]):
                                cv2.drawMarker(thermal_frame, (x+thermal_startX+10, y+thermal_startY+10), (0, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=5, thickness=1)
                        except:
                            max_coordinate = (int(thermal_max_index[0][0]), int(thermal_max_index[1][0]))
                            cv2.drawMarker(thermal_frame, (max_coordinate[0]+thermal_startX+10, max_coordinate[1]+thermal_startY+10), (0, 255, 0), markerType=cv2.MARKER_SQUARE, markerSize=5, thickness=1)

                    ## draw vertical line on thermal image
                    cv2.line(thermal_frame, (thermal_img_Cx, 0), (thermal_img_Cx, thermal_frame.shape[0]) , (0, 255, 0), 2)

                    ## drawing center of bounding box
                    cv2.circle(thermal_frame, (thermal_face_Cx, thermal_face_Cy), 2, (0, 255, 0), -1)

                ## draw bounding box
                cv2.rectangle(normal_frame, (startX, startY), (endX, endY), (0,255,0), 2)
                cv2.rectangle(thermal_frame, (thermal_startX, thermal_startY), (thermal_endX, thermal_endY), (0,255,0), 2)

                cv2.putText(normal_frame, text, (startX, name_txt_y_pose), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                cv2.putText(thermal_frame, text, (thermal_startX, thermal_txt_y_pose), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                print(f"[INFO] current person: {text}")

        print(f"[INFO] Last frame size: {normal_frame.shape}")

        ## check frame
        sec = currTime - prevTime
        prevTime = currTime
        fps = 1 / sec
        fps_str_format = "FPS: {%.2f}" % fps 
        print("[INFO] approx. current FPS: {:.2f}".format(fps))
        cv2.putText(normal_frame, fps_str_format, (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Normal Frame", normal_frame)
        cv2.imshow("Thermal Frame", thermal_frame)

        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            break

    # vs.stream.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-S1", "--source1", default="rtsp://admin:kmjeon3121@192.168.0.108:554/cam/realmonitor?channel=1&subtype=0", type=str, help="Path to the rtsp address of webcam or the path of the video.")
    parser.add_argument("-S2", "--source2", default="rtsp://admin:kmjeon3121@192.168.0.108:554/cam/realmonitor?channel=2&subtype=0", type=str, help="Path to the rtsp address of webcam or the path of the video.")
    parser.add_argument('--embeddings_path', default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank/embeddings_info_MobileFaceNet_model-y1-test2.pickle", help="path to the embedding info pickle file.")
    parser.add_argument('--encoded_label_path', default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/embedding_classifier/pytorch_embedding_classifier/label.pickle", help="path to the embedding info pickle file.")
    parser.add_argument("--classifier_path", default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/embedding_classifier/pytorch_embedding_classifier/embedding_classifier.pth")
    parser.add_argument('--face_detect_model_path', default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/face_detector/mobilenet_retinaface/mnet.25', type=str, help='path of the face detection model.')
    parser.add_argument('--detector_epoch', default=0, type=int, help='epoch of the detector.')
    parser.add_argument('--embedding_model_path', default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/face_embedding_extractor/MobileFaceNet_model-y1-test2/model', type=str, help='path of the model extracing embeddings.')
    parser.add_argument('--embedding_epoch', default=0, type=int, help='epoch of the embedding model.')
    parser.add_argument("--resize_width", default=480, type=int, help="resize width.")
    parser.add_argument("--resize_height", default=360, type=int, help="resize width.")
    parser.add_argument("--frame_flip", default=False, type=str2bool, help="Flip frame or not.")
    parser.add_argument("--gpu_id", default='cpu', help="GPU ID or CPU")
    parser.add_argument('--image_size_for_align', default='112,112', type=str, help="image size for crop.")
    parser.add_argument('--det_threshold', default=0.8, type=float, help="detection threshold.")
    parser.add_argument('--frame_num_for_detection', default=5, type=int)
    parser.add_argument('--proba_threshold', default=0.6)
    parser.add_argument('--cosine_threshold', default=0.6)

    args = vars(parser.parse_args())

    #adjustable variables
    scale_candidate = [240, 480]
    # scale_candidate = [120, 240]

    #Do detection and embedding comparison.
    main(args, scale_candidate)
 
    