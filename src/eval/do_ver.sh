
#python -u verification.py --gpu 0 --data-dir /opt/jiaguo/faces_vgg_112x112 --image-size 112,112 --model '../../model/softmax1010d3-r101-p0_0_96_112_0,21|22|32' --target agedb_30
# python -u verification.py --gpu=0 --data-dir='/home/gbkim/gb_dev/insightface_MXNet/insightface/datasets/faces_emore' --model='/home/gbkim/gb_dev/insightface_MXNet/insightface/models/model-y1-test2/model,0' --target 'agedb_30' --batch-size 32

python verification.py --gpu=0 --data-dir='/home/gbkim/gb_dev/insightface_MXNet/insightface/datasets/faces_emore' --model='/home/gbkim/gb_dev/insightface_MXNet/insightface/models/model-r100-ii/model,0' --target 'cfp_fp' --batch-size 32