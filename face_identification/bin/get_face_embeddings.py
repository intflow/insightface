"""
*Initial writer : GBKim
*Date           : 29.May.2020
*Description    :
    - Get embeddings from facebank for identification.
*logs           :
    - GBKim 05.29.2020: Initial coding.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'utils'))

import cv2
import pickle
import argparse
import numpy as np
from imutils import paths
from utils.face_detection_and_align import AnalyzeFace


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


def main(args, scale_candidate):
    #Grab the paths of images
    print("[INFO] Grabbing the paths of iamges...")
    imagePaths = list(paths.list_images(args['dataset_path']))

    ## Load detection model and embedding extractor
    model = AnalyzeFace(args, scale_candidate)

    ## Initialize lists of embeddings and names
    embeddings_list = []
    names_list = []

    ## Initialize the total number of processed faces
    total_face = 0

    for (i, each_img_path) in enumerate(imagePaths):
        name = each_img_path.split(os.path.sep)[-2]
        print("[INFO] processing on {0}-th/{1}. The name of the image is: {2}".format(i+1, len(imagePaths), name))

        #Load image
        img = cv2.imread(each_img_path)
        
        #Detect face and get embedding information
        alligned_img, face_bbox, landmarks = model.get_input(img)
        embedding_information = model.get_feature(alligned_img)

        #Append the name of the person + corresponding face embedding to their respective list
        embeddings_list.append(embedding_information)
        names_list.append(name)
        total_face += 1
    print("[INFO] {} images has been processed to get embedding information.".format(total_face))

    ## Save embeddings_list and names_list
    data = {"embeddings": embeddings_list, "names": names_list}
    f = open(args['save_path_'], 'wb')
    f.write(pickle.dumps(data))
    f.close()
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    ## Argumetns for data
    parser.add_argument("--dataset_path", default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank")
    parser.add_argument("--save_path_", default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank/embeddings_info2.pickle")

    ## Arguments for RetinaFace
    parser.add_argument('--gpu_id', default=0, type=int, help="GPU ID.")
    parser.add_argument('--image_size_for_align', default='112,112', type=str, help="image size for crop.")
    parser.add_argument('--face_detect_model_path', default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/retinaface-R50/R50', type=str, help='path of the face detection model.')
    parser.add_argument('--detector_epoch', default=0, type=int, help='epoch of the detector.')
    parser.add_argument('--embedding_model_path', default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/MobileFaceNet_model-y1-test2/model', type=str, help='path of the model extracing embeddings.')
    parser.add_argument('--embedding_epoch', default=0, type=int, help='epoch of the embedding model.')
    parser.add_argument('--det_threshold', default=0.8, type=float, help="detection threshold.")
    parser.add_argument('--frame_flip', default=False, type=str2bool, help='Flip frame or not.')
    parser.add_argument('--resize_width', default=112, type=int, help='resize width.')
    parser.add_argument('--resize_height', default=112, type=int, help='resize height.')
    args = vars(parser.parse_args())

    #adjustable variables
    scale_candidate = [270, 480]

    #Do face detection and get embeddings information.
    main(args, scale_candidate)





