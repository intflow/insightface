import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append('..')
## Add directy path for importing packages

import cv2
import argparse
import numpy as np
from PIL import Image
import face_model

parser = argparse.ArgumentParser(description='face model test')
# general
parser.add_argument('--image-size', default='112,112', help='')
parser.add_argument('--model', default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/LResNet50E-I_model-r50-am-lfw/model, 0', help='path to load model.')
parser.add_argument('--ga-model', default='', help='path to load model.')
parser.add_argument('--gpu', default=0, type=int, help='gpu id')
parser.add_argument('--det', default=0, type=int, help='mtcnn option, 1 means using R+O, 0 means detect from begining')
parser.add_argument('--flip', default=0, type=int, help='whether do lr flip aug')
parser.add_argument('--threshold', default=1.24, type=float, help='ver dist threshold')
args = parser.parse_args()

if __name__ == "__main__":
    model = face_model.FaceModel(args)
    img = cv2.imread('/home/gbkim/gb_dev/insightface_MXNet/insightface/deploy/Tom_Hanks_54745.png')
    # img = Image.open('/home/gbkim/gb_dev/insightface_MXNet/insightface/deploy/Tom_Hanks_54745.png')
    # img = np.array(img)
    img = model.get_input(img)
    f1 = model.get_feature(img)

    img2 = cv2.imread('/home/gbkim/gb_dev/insightface_MXNet/insightface/deploy/Tom_Hanks_test.png')
    # img2 = Image.open('/home/gbkim/gb_dev/insightface_MXNet/insightface/deploy/Tom_Hanks_test.png')
    img2 = model.get_input(img2)
    f2 = model.get_feature(img2)
    dist = np.sum(np.square(f1-f2))
    print("dist: ", dist)
    sim = np.dot(f1, f2.T)
    print("sim: ", sim)

    

