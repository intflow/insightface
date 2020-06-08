import insightface
import urllib
import urllib.request
import cv2
import numpy as np

def url_to_image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    # return the image
    return image

url = 'https://github.com/deepinsight/insightface/raw/master/deploy/Tom_Hanks_54745.png'
img = url_to_image(url)

# model = insightface.model_zoo.get_model('arcface_r100_v1')
model = insightface.model_zoo.get_model('arcface_r100_v1')

model.prepare(ctx_id = 0)
print(model)
import pdb; pdb.set_trace()

emb = model.get_embedding(img)

print(emb)