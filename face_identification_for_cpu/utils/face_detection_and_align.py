"""
*Initial writer : GBKim
*Date           : 28.May.2020
*Description    :
    - (28.May.2020) This code will help you to load model for embedding and to get embedding information.
*logs           :
    - GBKim 05.28.2020: Initial coding.
"""

import sys
import os
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import cv2
import sklearn.preprocessing as skl_preprocessing
import numpy as np
import mxnet as mx
import imutils
from .retinaface import RetinaFace
from . import face_image
from . import face_preprocess


def load_embedding_model(model_str, epoch, ctx, image_size, layer):
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

    prefix = model_str
    epoch = int(epoch)
    
    # print('[INFO] loading model...{} with {}-th epoch.'.format(prefix, epoch))

    sym, arg_params, aux_params = mx.model.load_checkpoint(model_str, epoch)
    all_layers = sym.get_internals()
    sym = all_layers[layer + '_output']

    model = mx.mod.Module(symbol=sym, context=mx.gpu(ctx), label_names=None)
    model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
    model.set_params(arg_params, aux_params)

    return model


class AnalyzeFace:
    def __init__(self, args, scale_candidate):
        self.scale_candidate = scale_candidate
        self.args = args
        ctx = mx.gpu(self.args['gpu_id'])
        vec_image_size = self.args['image_size_for_align'].split(',')
        assert len(vec_image_size)==2
        self.image_size = (int(vec_image_size[0]), int(vec_image_size[1]))
    
        self.face_detector = RetinaFace(prefix=self.args['face_detect_model_path'], epoch=self.args['detector_epoch'], ctx_id=self.args['gpu_id'], network='net3')
        print("[INFO] RetinaFace has been loaded as a face_detector...")

        self.embedding_model = load_embedding_model(model_str=self.args['embedding_model_path'], epoch=self.args['embedding_epoch'], ctx=self.args['gpu_id'], image_size=self.image_size, layer='fc1')
        print("[INFO] embedding model has been loaded as a embedding extractor...")

    def get_input(self, face_img):
        resized_img = imutils.resize(face_img, width=self.args['resize_width'])
        print(f"[INFO] Image size: {resized_img.shape}")
        
        target_size = self.scale_candidate[0]
        max_size = self.scale_candidate[1]

        img_width = resized_img.shape[0]
        img_height = resized_img.shape[1]

        im_size_min = np.min(resized_img.shape[0:2])
        im_size_max = np.max(resized_img.shape[0:2])
        im_scale = float(target_size) / float(im_size_min)

        if np.round(im_scale * im_size_max) > max_size: # prevent bigger axis from being more than max_size
            im_scale = float(max_size) / float(im_size_max)
        print(f"[INFO] image scale: {im_scale}")
        self.scales = [im_scale]

        face_bbox, landmarks = self.face_detector.detect(resized_img, self.args["det_threshold"], scales=self.scales, do_flip=self.args['frame_flip'])

        if face_bbox is None:
            return None

        if face_bbox is not None:
            print("Find, {} face_bbox.".format(face_bbox.shape[0]))

            for i in range(face_bbox.shape[0]):
                box = face_bbox[i].astype(np.int)
                print(f"[INFO] Information {i+1}-th face box: {box}")
                color1 = (0,0,255)

                if landmarks is not None:
                    each_landmark = landmarks[i].astype(np.int)

                # import matplotlib.pyplot as plt
                # plt.figure('resized_img') # chk_point1
                # plt.imshow(resized_img)
                
                processed_img = face_preprocess.preprocess(resized_img, bbox=box, landmark=each_landmark, image_size=self.args['image_size_for_align'])
                processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)

                
                # plt.figure('processed_img') # chk_point2
                # plt.imshow(processed_img)
                # plt.show()

                aligned = np.transpose(processed_img, (2, 0 ,1))

        return aligned, face_bbox, landmarks


    def get_feature(self, aligned_img):
        input_img = np.expand_dims(aligned_img, axis=0)
        data = mx.nd.array(input_img)
        db = mx.io.DataBatch(data=(data,))
        self.embedding_model.forward(db, is_train=False)
        embedding = self.embedding_model.get_outputs()[0].asnumpy()
        embedding = skl_preprocessing.normalize(embedding).flatten()
        return embedding







    
