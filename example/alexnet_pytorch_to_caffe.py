import sys
sys.path.insert(0,'.')
import torch
from torch.autograd import Variable
from torchvision.models.alexnet import alexnet
import pytorch_to_caffe

if __name__=='__main__':
    sys.stdout = open("./debug_files/log.txt", "w+")
    print("alexnet experiment \n")
    name='alexnet'
    net=alexnet(True)
    input=Variable(torch.ones([1,3,226,226]))
    pytorch_to_caffe.trans_net(net,input,name)
    pytorch_to_caffe.save_prototxt('./example_models/{}.prototxt'.format(name))
    pytorch_to_caffe.save_caffemodel('./example_models/{}.caffemodel'.format(name))
    # sys.stdout.flush()
