import mxnet as mx

model = mx.model.load_checkpoint("/home/gbkim/gb_dev/insightface_MXNet/insightface/models/MobileFaceNet_model-y1-test2/model", 0)

# print(model)

print(dir(model))

model2 = mx.model.load_checkpoint("/home/gbkim/.insightface/models/arcface_r100_v1/model", 0)

print(model2)
print(dir(model2))


model2 = model2.prepare(ctx_id = 0)

print(model2)