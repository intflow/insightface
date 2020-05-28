"""
*Initial writer : GBKim
*Date           : 29.May.2020
*Description    :
    - face identification for an image.
*logs           :
    - GBKim 05.29.2020: Initial coding.
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import cv2
import numpy as np
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils.get_face_embedding import AnalyzeFace

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
    """main function to do face detection, get embeddings and comparison between two images.

    Args:
        args (argparser)
        scales (list or tuple of two elements)
    """

    ## Load models
    model = AnalyzeFace(args, scale_candidate)

    img1 = cv2.imread('/home/gbkim/gb_dev/insightface_MXNet/insightface/deploy/Tom_Hanks_54745.png')
    img1 = model.get_input(img1)
    f1 = model.get_feature(img1)
    
    img2 = cv2.imread('/home/gbkim/gb_dev/insightface_MXNet/insightface/deploy/Tom_Hanks_test.png')
    img2 = model.get_input(img2)
    f2 = model.get_feature(img2)

    dist = np.sum(np.square(f1 - f2))
    print("dist: ", dist)
    similarity = np.dot(f1, f2.T)
    print('similairty: ', similarity)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='argparser for face identification.')
    parser.add_argument('--gpu_id', default=0, type=int, help="GPU ID.")
    parser.add_argument('--image_size_for_align', default='112,112', type=str, help="image size for crop.")
    parser.add_argument('--face_detect_model_path', default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/retinaface-R50/R50', type=str, help='path of the face detection model.')
    parser.add_argument('--detector_epoch', default=0, type=int, help='epoch of the detector.')
    parser.add_argument('--embedding_model_path', default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/LResNet50E-I_model-r50-am-lfw/model', type=str, help='path of the model extracing embeddings.')
    parser.add_argument('--embedding_epoch', default=0, type=int, help='epoch of the embedding model.')
    parser.add_argument('--det_threshold', default=0.8, type=float, help="detection threshold.")
    parser.add_argument('--frame_flip', default=False, type=str2bool, help='Flip frame or not.')
    parser.add_argument('--resize_width', default=480, type=int, help='resize width.')
    parser.add_argument('--resize_height', default=270, type=int, help='resize height.')
    args = vars(parser.parse_args())

    #adjustable variables
    scale_candidate = [270, 480]

    #Do detection and embedding comparison.
    main(args, scale_candidate)
