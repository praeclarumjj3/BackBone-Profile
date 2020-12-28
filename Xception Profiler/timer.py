""" 
Creates an Xception Model as defined in:
Francois Chollet
Xception: Deep Learning with Depthwise Separable Convolutions
https://arxiv.org/pdf/1610.02357.pdf
This weights ported from the Keras implementation. Achieves the following performance on the validation set:
Loss:0.9173 Prec@1:78.892 Prec@5:94.292
REMEMBER to set your image size to 3x299x299 for both test and validation
normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                  std=[0.5, 0.5, 0.5])
The resize parameter of the validation transform should be 333, and make sure to center crop at 299x299
"""
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch
import numpy as np
import argparse
import os
from xception import xception1

__all__ = ['xception']

model_urls = {
    'xception':'https://www.dropbox.com/s/1hplpzet9d7dv29/xception-c0a72b38.pth.tar?dl=1'
}

xstart = torch.cuda.Event(enable_timing=True)
xend = torch.cuda.Event(enable_timing=True)

tstart = torch.cuda.Event(enable_timing=True)
tend = torch.cuda.Event(enable_timing=True)

layers = {"Conv2D-1": [["e"],[0]],
        "BN+ReLU-1": [["e"],[0]],
        "Conv2D-2": [["e"],[0]],
        "BN+ReLU-2": [["e"],[0]],
        "Block-1": [["e"],[0]],
        "Block-2": [["e"],[0]],
        "Block-3": [["e"],[0]],
        "Middle_Blocks": [],
        "Block-12": [["e"],[0]],
        "SepConv2D-3": [["e"],[0]],
        "BN+ReLU-3": [["e"],[0]],
        "SepConv2D-4": [["e"],[0]],
        "BN+ReLU-4": [["e"],[0]],
        "AvgPool": [["e"],[0]],
        "FC": [["e"],[0]]}

def add_time(key,value,i=None):
    if key == "Middle_Blocks":
        if len(layers[key][i][0]) is 1:
            layers[key][i][0].append(value[0])
        layers[key][i][1].append(value[1])
    else:
        if len(layers[key][0]) is 1:
            layers[key][0].append(value[0])
        layers[key][1].append(value[1])

class SeparableConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,dilation=1,bias=False):
        super(SeparableConv2d,self).__init__()

        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size,stride,padding,dilation,groups=in_channels,bias=bias)
        self.pointwise = nn.Conv2d(in_channels,out_channels,1,1,0,1,1,bias=bias)
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self,in_filters,out_filters,reps,strides=1,start_with_relu=True,grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides!=1:
            self.skip = nn.Conv2d(in_filters,out_filters,1,stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip=None
        
        self.relu = nn.ReLU(inplace=True)
        rep=[]

        filters=in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters,filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(filters))
        
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters,out_filters,3,stride=1,padding=1,bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3,strides,1))
        self.rep = nn.Sequential(*rep)

    def forward(self,inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x+=skip
        return x


class Xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """
    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()

        
        self.num_classes = num_classes

        ## entry_flow
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(32,64,3,bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.block1=Block(64,128,2,2,start_with_relu=False,grow_first=True)
        self.block2=Block(128,256,2,2,start_with_relu=True,grow_first=True)
        self.block3=Block(256,728,2,2,start_with_relu=True,grow_first=True)

        ## middle_flow 
        self.middle_blocks = []
        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))
        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))
        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))
        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))

        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))
        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))
        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))
        self.middle_blocks.append(Block(728,728,3,1,start_with_relu=True,grow_first=True))

        self.middle = nn.ModuleList(self.middle_blocks)
        for i in range(len(self.middle_blocks)):
            layers["Middle_Blocks"].append([["e"],[0]])

        ## exit_flow
        self.block12=Block(728,1024,2,2,start_with_relu=True,grow_first=False)
        
        self.conv3 = SeparableConv2d(1024,1536,3,1,1)
        self.bn3 = nn.BatchNorm2d(1536)

        #do relu here
        self.conv4 = SeparableConv2d(1536,2048,3,1,1)
        self.bn4 = nn.BatchNorm2d(2048)

        self.fc = nn.Linear(2048, num_classes)

        #------- init weights --------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #-----------------------------

    def forward(self, x):
        
        shape = list(x.shape)
        xstart.record()
        x = self.conv1(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("Conv2D-1",("with kernel_size=3, stride=2 and input shape: {}".format(shape),xstart.elapsed_time(xend)))

        shape = list(x.shape)
        xstart.record()
        x = self.bn1(x)
        x = self.relu(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("BN+ReLU-1",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        

        shape = list(x.shape)
        xstart.record()
        x = self.conv2(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("Conv2D-2",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))

        shape = list(x.shape)
        xstart.record()
        x = self.bn2(x)
        x = self.relu(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("BN+ReLU-2",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        
        shape = list(x.shape)
        xstart.record()
        x = self.block1(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("Block-1",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        
        shape = list(x.shape)
        xstart.record()
        x = self.block2(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("Block-2",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        
        shape = list(x.shape)
        xstart.record()
        x = self.block3(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("Block-3",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))

        for i in range(len(self.middle_blocks)):
            shape = list(x.shape)
            xstart.record()
            x = self.middle[i](x)
            xend.record()
            torch.cuda.synchronize()
            add_time("Middle_Blocks",("at position {} with input shape: {}".format(i+1,shape),xstart.elapsed_time(xend)),i)
            
           
        shape = list(x.shape)
        xstart.record()
        x = self.block12(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("Block-12",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        
        shape = list(x.shape)
        xstart.record()
        x = self.conv3(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("SepConv2D-3",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        
        shape = list(x.shape)
        xstart.record()
        x = self.bn3(x)
        x = self.relu(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("BN+ReLU-3",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        
        shape = list(x.shape)
        xstart.record()
        x = self.conv4(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("SepConv2D-4",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))
        
        shape = list(x.shape)
        xstart.record()
        x = self.bn4(x)
        x = self.relu(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("BN+ReLU-4",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))

        shape = list(x.shape)
        xstart.record()
        x = F.adaptive_avg_pool2d(x, (1, 1))
        xend.record()
        torch.cuda.synchronize()
        add_time("AvgPool",("with input shape: {}".format(shape),xstart.elapsed_time(xend)))

        x = x.view(x.size(0), -1)

        shape = list(x.shape)
        xstart.record()
        x = self.fc(x)
        xend.record()
        torch.cuda.synchronize()
        add_time("FC",("with dim=2048 and input shape: {}".format(shape),xstart.elapsed_time(xend)))

        return x
    
def xception(pretrained=False,**kwargs):
    """
    Construct Xception.
    """

    model = Xception(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['xception']))
    return model


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train_overexposure")
    parser.add_argument("--i", default="cityscapes",
    help="input dataset from ['cityscapes', 'pascal']")
    parser.add_argument("--w", default="yes",
    help="To calculate layerwise or total inference time")
    args =  parser.parse_args()

    device = torch.device("cuda")

    if args.i == 'pascal':
        img = torch.randn(1, 3, 500, 334).to(device)
    else:
        img = torch.randn(1, 3, 1024, 2048).to(device)

    model = xception(False).to(device)
    warm_model = xception1(False).to(device)
    model_whole = xception1(False).to(device)
    if not os.path.exists('xception_stats'):
        os.makedirs('xception_stats')
    
    # Uncomment to print the model structure
    # print(model)
    
    warm_iter = 25
    with torch.no_grad():
        for i in range(warm_iter):
            warm_model(img)
    iter = 300
    times = []
    
    if args.w == "yes":
        with torch.no_grad():
            for i in range(iter):
                shape_input = list(img.shape)
                tstart.record()
                model_whole(img)
                tend.record()
                torch.cuda.synchronize()
                times.append(tstart.elapsed_time(tend))
    else:
        with torch.no_grad():
            for i in range(iter):
                shape_input = list(img.shape)
                tstart.record()
                model(img)
                tend.record()
                torch.cuda.synchronize()
                times.append(tstart.elapsed_time(tend))
    
    if args.w == "yes":
        pass
    
    else:
        ## Conv2D-1
        m = np.mean(layers["Conv2D-1"][1][1:])
        s = np.std(layers["Conv2D-1"][1][1:])
        print("Average Execution Time for Conv2D-1 {} is {} ms with std = {} ms".format(layers["Conv2D-1"][0][1],round(m,3),round(s,3)))

        ## BN+ReLU-1
        m = np.mean(layers["BN+ReLU-1"][1][1:])
        s = np.std(layers["BN+ReLU-1"][1][1:])
        print("Average Execution Time for BN+ReLU-1 {} is {} ms with std = {} ms".format(layers["BN+ReLU-1"][0][1],round(m,3),round(s,3)))

        ## Conv2D-2
        m = np.mean(layers["Conv2D-2"][1][1:])
        s = np.std(layers["Conv2D-2"][1][1:])
        print("Average Execution Time for Conv2D-2 {} is {} ms with std = {} ms".format(layers["Conv2D-2"][0][1],round(m,3),round(s,3)))

        ## BN+ReLU-2
        m = np.mean(layers["BN+ReLU-2"][1][1:])
        s = np.std(layers["BN+ReLU-2"][1][1:])
        print("Average Execution Time for BN+ReLU-2 {} is {} ms with std = {} ms".format(layers["BN+ReLU-2"][0][1],round(m,3),round(s,3)))

        ## Block-1
        m = np.mean(layers["Block-1"][1][1:])
        s = np.std(layers["Block-1"][1][1:])
        print("Average Execution Time for Block-1 {} is {} ms with std = {} ms".format(layers["Block-1"][0][1],round(m,3),round(s,3)))

        ## Block-2
        m = np.mean(layers["Block-2"][1][1:])
        s = np.std(layers["Block-2"][1][1:])
        print("Average Execution Time for Block-2 {} is {} ms with std = {} ms".format(layers["Block-2"][0][1],round(m,3),round(s,3)))

        ## Block-3
        m = np.mean(layers["Block-3"][1][1:])
        s = np.std(layers["Block-3"][1][1:])
        print("Average Execution Time for Block-3 {} is {} ms with std = {} ms".format(layers["Block-3"][0][1],round(m,3),round(s,3)))


        ## Blocks
        for i in range(len(layers["Middle_Blocks"])):
            m = np.mean(layers["Middle_Blocks"][i][1][1:])
            s = np.std(layers["Middle_Blocks"][i][1][1:])
            print("Average Execution Time for Middle Block {} is {} ms with std = {} ms".format(layers["Middle_Blocks"][i][0][1],round(m,3),round(s,3)))

        ## Block-12
        m = np.mean(layers["Block-12"][1][1:])
        s = np.std(layers["Block-12"][1][1:])
        print("Average Execution Time for Block-12 {} is {} ms with std = {} ms".format(layers["Block-12"][0][1],round(m,3),round(s,3)))

        ## Conv2D-3
        m = np.mean(layers["SepConv2D-3"][1][1:])
        s = np.std(layers["SepConv2D-3"][1][1:])
        print("Average Execution Time for SepConv2D-3 {} is {} ms with std = {} ms".format(layers["SepConv2D-3"][0][1],round(m,3),round(s,3)))

        ## BN+ReLU-3
        m = np.mean(layers["BN+ReLU-3"][1][1:])
        s = np.std(layers["BN+ReLU-3"][1][1:])
        print("Average Execution Time for BN+ReLU-3 {} is {} ms with std = {} ms".format(layers["BN+ReLU-3"][0][1],round(m,3),round(s,3)))

        ## Conv2D-4
        m = np.mean(layers["SepConv2D-4"][1][1:])
        s = np.std(layers["SepConv2D-4"][1][1:])
        print("Average Execution Time for SepConv2D-4 {} is {} ms with std = {} ms".format(layers["SepConv2D-4"][0][1],round(m,3),round(s,3)))

        ## BN+ReLU-4
        m = np.mean(layers["BN+ReLU-4"][1][1:])
        s = np.std(layers["BN+ReLU-4"][1][1:])
        print("Average Execution Time for BN+ReLU-4 {} is {} ms with std = {} ms".format(layers["BN+ReLU-4"][0][1],round(m,3),round(s,3)))

        ## AvgPool
        m = np.mean(layers["AvgPool"][1][1:])
        s = np.std(layers["AvgPool"][1][1:])
        print("Average Execution Time for AvgPool {} is {} ms with std = {} ms".format(layers["AvgPool"][0][1],round(m,3),round(s,3)))

        ## FC
        m = np.mean(layers["FC"][1][1:])
        s = np.std(layers["FC"][1][1:])
        print("Average Execution Time for FC {} is {} ms with std = {} ms\n".format(layers["FC"][0][1],round(m,3),round(s,3)))

    ## Total
    m = np.sum(times)/iter
    s = np.std(times)
    print("Average Execution Time for Xception with input shape: {} is {} ms with std = {} ms".format(shape_input,round(m,3),round(s,3)))
    
    torch.cuda.empty_cache()
