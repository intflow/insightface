import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'utils'))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(os.path.dirname(__file__))), 'architecture'))

import numpy as np
import argparse
import pickle

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from keras.models import load_model
from architecture.embedding_learner_keras import SoftMax 
import matplotlib.pyplot as plt


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


def main(args):
    ## Load face embeddings
    embedding_data = pickle.loads(open(args['embeddings_path'], 'rb').read())
    print("[INFO] embedding data has been loaded...")

    ## Encode the labels
    sklearn_label_encoder = LabelEncoder()
    labels = sklearn_label_encoder.fit_transform(embedding_data['names'])
    num_classes = len(np.unique(labels))
    labels = labels.reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(categories='auto')
    labels = one_hot_encoder.fit_transform(labels).toarray()
    
    embeddings = np.array(embedding_data['embeddings'])

    ## Initialize classifier arguments
    BATCH_SIZE = args['batch_size']
    EPOCHS = args['epochs']
    input_shape = embeddings.shape[1]

    # Build sofmax classifier
    softmax = SoftMax(input_shape=(input_shape,), num_classes=num_classes)
    model = softmax.build()

    # Create KFold
    cv = KFold(n_splits = 5, random_state = 31, shuffle=True)
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    # Train
    for train_idx, valid_idx in cv.split(embeddings):
        X_train, X_val, y_train, y_val = embeddings[train_idx], embeddings[valid_idx], labels[train_idx], labels[valid_idx]

        his = model.fit(x=X_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_data=(X_val, y_val))

        print(his.history['accuracy'])

        history['acc'] += his.history['accuracy']
        history['val_acc'] += his.history['val_accuracy']
        history['loss'] += his.history['loss']
        history['val_loss'] += his.history['val_loss']


    # write the face recognition model to output
    model.save(args['model_save_path'])
    f = open(args["encoded_label_save_path"], "wb")
    f.write(pickle.dumps(sklearn_label_encoder))
    f.close()

    # Plot
    plt.figure(1)

    # Summary history for accuracy
    plt.subplot(211)
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # Summary history for loss
    plt.subplot(212)
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(args['figure_save_path'])
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings_path', default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank/embeddings_info.pickle", help="path to the embedding info pickle file.")
    parser.add_argument('--model_save_path', default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank/embedding_classifier.h5", help="path of the model to be saved.")
    parser.add_argument('--encoded_label_save_path', default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank/label.pickle", help="path of the label encoder to be saved.")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size for model training.")
    parser.add_argument("--epochs", default=50, type=int, help="Epochs for training.")
    parser.add_argument("--figure_save_path", default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank/result_figure.png")

    args = vars(parser.parse_args())

    main(args)
