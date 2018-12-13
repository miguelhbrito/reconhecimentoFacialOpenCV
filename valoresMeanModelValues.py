import caffe
import numpy as np

blob = caffe.proto.caffe_pb2.BlobProto()
with open('mean.binaryproto','rb') as f:
    blob.ParseFromString(f.read())
    data = np.array(blob.data).reshape([blob.channels, blob.height, blob.width])
    print (np.mean(data[0]), np.mean(data[1]), np.mean(data[2]))
