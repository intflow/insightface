## Load Packages
import sys, os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))), 'utils'))
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))), 'architecture'))
import cv2
import torch
use_cuda = torch.cuda.is_available()
import pickle
import argparse
import numpy as np
import torchsummary
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader
from architecture.embedding_learner_pytorch import embedding_classifier


## Define custom dataloader
def custom_dataloader(x, y, batch_size):
    if batch_size > len(y):
        batch_size = len(y)

    tensor_x = torch.tensor(x)
    tensor_y = torch.tensor(y, dtype=torch.long)

    if use_cuda:
        tensor_x = tensor_x.cuda()
        tensor_y = tensor_y.cuda()

    # one-hot label to class
    tensor_y = torch.argmax(tensor_y, dim=1)

    my_tensor_dataset = TensorDataset(tensor_x, tensor_y)
        
    return DataLoader(my_tensor_dataset, batch_size=batch_size, shuffle=True)


## Layer weight initialization
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

## Do train
def main(args):
    ## Set epoch and batch_size
    EPOCHS = args['epochs']
    BATCH_SIZE = args['batch_size']

    ##Load face embeddings
    embedding_data = pickle.loads(open(args['input_embedding_path'], 'rb').read())
    print("[INFO] embedding data has been loaded...")
    input_embeddings = np.array(embedding_data['embeddings'])

    ## Encode the labels
    sklearn_label_encoder = LabelEncoder()
    labels = sklearn_label_encoder.fit_transform(embedding_data['names'])
    num_classes = len(np.unique(labels))
    labels = labels.reshape(-1, 1)
    one_hot_encoder = OneHotEncoder(categories='auto')
    labels = one_hot_encoder.fit_transform(labels).toarray()

    ## Set model
    model = embedding_classifier(input_shape=input_embeddings.shape[1], num_classes=num_classes)
    model.apply(weights_init)
    torchsummary.summary(model, (input_embeddings.shape[1], ))

    ## define loss
    criterion = torch.nn.CrossEntropyLoss()

    ## define optimizer
    optimizer = torch.optim.Adam(model.parameters())

    ## Set train Data Loader
    train_data_loader = custom_dataloader(x=input_embeddings, y=labels, batch_size=BATCH_SIZE)

    ## Initialize list of training information 
    history = {'train_acc': [], 'train_loss': []}

    ## Do training
    for epoch in range(EPOCHS):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        for i, data in enumerate(train_data_loader):
            x, y = data

            ## grad init
            optimizer.zero_grad()
            
            ## forward propagation
            model_output = model(x)

            ## calculate loss
            loss = criterion(model_output, y)

            ## back propagation
            loss.backward()

            ## weight update
            optimizer.step()

            ## calculate trainig loss and accuracy
            train_loss += loss.item()
            train_preds = torch.argmax(model_output, dim=1)
            train_acc += train_preds.eq(y).float().mean().cpu().numpy()

            ## delete some variables for memory issue
            del loss
            del model_output

                
        epoch_train_accuracy = train_acc / len(train_data_loader)
        epoch_train_loss = train_loss / len(train_data_loader)

        history['train_acc'].append(epoch_train_accuracy)
        history['train_loss'].append(epoch_train_loss)


        print("epoch: {}/{} | training loss: {:.4f} | training acc: {:.4f}".format(epoch+1, EPOCHS, epoch_train_loss, epoch_train_accuracy))

        # train_loss = 0.0
        # train_acc = 0.0
        # val_loss = 0.0
        # val_acc = 0.0
    
    ## Save the pytorch embedding classifier
    torch.save(model.state_dict(), args["model_save_path"])

    ## Save label encoder
    f = open(args["encoded_label_save_path"], "wb")
    f.write(pickle.dumps(sklearn_label_encoder))
    f.close()

    ## Plot performance figure
    fig, axes = plt.subplots(2, 1, constrained_layout=True)
    
    axes[0].plot(history['train_acc'])
    # axes[0].plot(history['val_acc'])
    axes[0].set_title('model accuracy')
    axes[0].set_ylabel('accuracy')
    axes[0].set_xlabel('epochs')
    axes[0].legend(['train'], loc='best')

    # Summary history for loss
    axes[1].plot(history['train_loss'])
    # axes[1].plot(history['val_loss'])
    axes[1].set_title('model loss')
    axes[1].set_ylabel('loss')
    axes[1].set_xlabel('epochs')
    axes[1].legend(['train'], loc='best')
    
    plt.savefig(args['figure_save_path'], dpi=300)
    plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_embedding_path", default='/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/face_bank/embeddings_info2.pickle')
    ap.add_argument("--epochs", default=500, type=int, help="Epochs for training.")
    ap.add_argument("--batch_size", default=4, type=int, help="Batch size for model training.")
    ap.add_argument('--model_save_path', default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/embedding_classifier/pytorch_embedding_classifier/temp/embedding_classifier.pth", help="path of the model to be saved.")
    ap.add_argument('--encoded_label_save_path', default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/embedding_classifier/pytorch_embedding_classifier/temp/label.pickle", help="path of the label encoder to be saved.")
    ap.add_argument("--figure_save_path", default="/home/gbkim/gb_dev/insightface_MXNet/insightface/face_identification/model/embedding_classifier/pytorch_embedding_classifier/temp/result_figure.png")

    args = vars(ap.parse_args())

    main(args)