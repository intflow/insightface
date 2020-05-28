"""
*Initial writer : GBKim
*Date           : Wed.05.27.2020
*Description    :
    - This code use insigtface which is installed from pypi 
*logs           :
    - GBKim 05.27.2020: Initial coding.
"""

import cv2
import numpy as np
import argparse
import utils
from datetime import datetime
from retinaface import RetinaFace


def str2bool(v):
    if isinstance(v, bool):
        return v
    elif v.lower() in ['yes', 'true', 't', 'y', '1']:
        return True
    elif v.lower() in ['no', 'false', 'f', 'n', '0']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main(args):
    
    ## Load model
    detector = RetinaFace(prefix=args['model_path'], epoch=args['epoch'], ctx_id=args['gpu_id'], network='net3')
    print("[INFO] RetinaFace loaded.")

    if args['update']:
        targets, names = prepare_facebank()



if __name__=='__main__':
    ## Parse arguments
    parser = argparse.ArgumentParser(description="Argparser for face identification.")
    parser.add_argument("--model_path", default="./model/retinaface-R50/R50", type=str, help="Path of the model.")
    parser.add_argument("--epoch", default=0, type=int, help="model's epoch.")
    parser.add_argument("--gpu_id", default=0, type=int, help="GPU ID.")
    args = vars(parser.parse_args())

    main(args)



    

