import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models.vgg import vgg11_bn
import pytorch_to_caffe

if __name__=='__main__':
    name='vgg11_bn'
    sys.stdout = open("./debug_files/{}_log.txt".format(name), "w+")
    net=vgg11_bn(True)
    input=Variable(torch.ones([1,3,224,224]))
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('./example_models/{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('./example_models/{}.caffemodel'.format(name))