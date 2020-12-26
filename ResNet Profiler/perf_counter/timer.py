import torch
from torch import Tensor
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
import argparse
import os
import torchvision.models as models
import numpy as np
import time

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

layers = {"Conv2D": [["e"],[0]],
        "BN": [["e"],[0]],
        "MaxPool": [["e"],[0]],
        "Block1": [["e"],[0]],
        "Block2": [["e"],[0]],
        "Block3": [["e"],[0]],
        "Block4": [["e"],[0]],
        "AvgPool": [["e"],[0]],
        "FC": [["e"],[0]],}

g_device = "cuda"


def add_time(key,value):
    if len(layers[key][0]) is 2:
        layers[key][0].append(value[0])
    layers[key][1].append(value[1])

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
    
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
       
        shape = list(x.shape)
        start = time.perf_counter()
        x = self.conv1(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("Conv2D",("with kernel_size=7, stride=2, padding=3 and input shape: {}".format(shape),(end-start)*1000))
        
        shape = list(x.shape)
        start = time.perf_counter()
        x = self.bn1(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("BN",("with input shape: {}".format(shape),(end-start)*1000))
        
        x = self.relu(x)
        
        shape = list(x.shape)
        start = time.perf_counter()
        x = self.maxpool(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("MaxPool",("with input shape: {}".format(shape),(end-start)*1000))

        shape = list(x.shape)
        start = time.perf_counter()
        x = self.layer1(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("Block1",("with input shape: {}".format(shape),(end-start)*1000))
        
        shape = list(x.shape)
        start = time.perf_counter()
        x = self.layer2(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("Block2",("with input shape: {}".format(shape),(end-start)*1000))

        shape = list(x.shape)
        start = time.perf_counter()
        x = self.layer3(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("Block3",("with input shape: {}".format(shape),(end-start)*1000))
        
        shape = list(x.shape)
        start = time.perf_counter()
        x = self.layer4(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("Block4",("with input shape: {}".format(shape),(end-start)*1000))
        
        shape = list(x.shape)
        start = time.perf_counter()
        x = self.avgpool(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("AvgPool",("with input shape: {}".format(shape),(end-start)*1000))
        
        x = torch.flatten(x, 1)
        
        shape = list(x.shape)
        start = time.perf_counter()
        x = self.fc(x)
        end = time.perf_counter()
        torch.cuda.synchronize()
        add_time("FC",(" with dimension=2048 and input shape: {}".format(shape),(end-start)*1000))
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(block: Type[Union[BasicBlock, Bottleneck]], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', block, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(block: Type[Union[BasicBlock, Bottleneck]], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', block, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(block: Type[Union[BasicBlock, Bottleneck]], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', block, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(block: Type[Union[BasicBlock, Bottleneck]], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', block, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(block: Type[Union[BasicBlock, Bottleneck]], pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', block, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Train_overexposure")
    parser.add_argument("--d", default=18, type=int,
    help="depth of resnet from [18,34,50,101,152]")
    parser.add_argument("--i", default="cityscapes",
    help="input dataset from ['cityscapes', 'pascal']")
    parser.add_argument("--w", default=True, type=bool,
    help="To calculate layerwise or total inference time")
    parser.add_argument("--device", default="cuda",
    help="cpu or cuda")
    args =  parser.parse_args()

    device = torch.device("cuda")

    if args.i == 'pascal':
        img = torch.randn(1, 3, 500, 334).to(device)
    else:
        img = torch.randn(1, 3, 1024, 2048).to(device)

    if args.d is 18:
        model = resnet18(BasicBlock,False).to(device)
        warm_model = models.resnet18().to(device)
        model_whole = models.resnet18().to(device)
        if not os.path.exists('resnet18_stats'):
            os.makedirs('resnet18_stats')
    elif args.d is 34:
        model = resnet34(BasicBlock,False).to(device)
        warm_model = models.resnet34().to(device)
        model_whole = models.resnet34().to(device)
        if not os.path.exists('resnet34_stats'):
            os.makedirs('resnet34_stats')
    elif args.d is 50:
        model = resnet50(Bottleneck,False).to(device)
        warm_model = models.resnet50().to(device)
        model_whole = models.resnet50().to(device)
        if not os.path.exists('resnet50_stats'):
            os.makedirs('resnet50_stats')
    elif args.d is 101:
        model = resnet101(Bottleneck,False).to(device)
        warm_model = models.resnet101().to(device)
        model_whole = models.resnet101().to(device)
        if not os.path.exists('resnet101_stats'):
            os.makedirs('resnet101_stats')
    elif args.d is 152:
        model = resnet152(Bottleneck,False).to(device)
        warm_model = models.resnet152().to(device)
        model_whole = models.resnet152().to(device)
    else:
        print("Please enter a valid depth!")
        exit()
    
    # Uncomment to print the model structure
    # print(model)
    
    warm_iter = 25
    with torch.no_grad():
        for i in range(warm_iter):
            warm_model(img)

    iter = 300
    times = []
    
    if args.w:
        with torch.no_grad():
            for i in range(iter):
                shape_input = list(img.shape)
                start = time.perf_counter()
                model_whole(img)
                end = time.perf_counter()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                times.append((end-start)*1000)
    else:
        with torch.no_grad():
            for i in range(iter):
                shape_input = list(img.shape)
                start = time.perf_counter()
                model_whole(img)
                end = time.perf_counter()
                if args.device == "cuda":
                    torch.cuda.synchronize()
                times.append((end-start)*1000)
    
    if not args.w:
        
        ## Conv2D
        m = np.mean(layers["Conv2D"][1][1:])
        s = np.std(layers["Conv2D"][1][1:])
        print("Average Execution Time for Conv2D {} is {} ms with std = {} ms".format(layers["Conv2D"][0][1],round(m,3),round(s,3)))

        ## BN
        m = np.mean(layers["BN"][1][1:])
        s = np.std(layers["BN"][1][1:])
        print("Average Execution Time for BN {} is {} ms with std = {} ms".format(layers["BN"][0][1],round(m,3),round(s,3)))

        ## MaxPool
        m = np.mean(layers["MaxPool"][1][1:])
        s = np.std(layers["MaxPool"][1][1:])
        print("Average Execution Time for MaxPool {} is {} ms with std = {} ms".format(layers["MaxPool"][0][1],round(m,3),round(s,3)))

        ## Block1
        m = np.mean(layers["Block1"][1][1:])
        s = np.std(layers["Block1"][1][1:])
        print("Average Execution Time for Block1 {} is {} ms with std = {} ms".format(layers["Block1"][0][1],round(m,3),round(s,3)))

        ## Block2
        m = np.mean(layers["Block2"][1][1:])
        s = np.std(layers["Block2"][1][1:])
        print("Average Execution Time for Block2 {} is {} ms with std = {} ms".format(layers["Block2"][0][1],round(m,3),round(s,3)))

        ## Block3
        m = np.mean(layers["Block3"][1][1:])
        s = np.std(layers["Block3"][1][1:])
        print("Average Execution Time for Block3 {} is {} ms with std = {} ms".format(layers["Block3"][0][1],round(m,3),round(s,3)))

        ## Block4
        m = np.mean(layers["Block4"][1][1:])
        s = np.std(layers["Block4"][1][1:])
        print("Average Execution Time for Block3 {} is {} ms with std = {} ms".format(layers["Block4"][0][1],round(m,3),round(s,3)))

        ## AvgPool
        m = np.mean(layers["AvgPool"][1][1:])
        s = np.std(layers["AvgPool"][1][1:])
        print("Average Execution Time for AvgPool {} is {} ms with std = {} ms".format(layers["AvgPool"][0][1],round(m,3),round(s,3)))

        ## FC
        m = np.mean(layers["FC"][1][1:])
        s = np.std(layers["FC"][1][1:])
        print("Average Execution Time for FC {} is {} ms with std = {} ms".format(layers["FC"][0][1],round(m,3),round(s,3)))

    ## Total
    m = np.sum(times)/iter
    s = np.std(times)
    print("Average Execution Time for ResNet-{} with input shape: {} is {} ms with std = {} ms".format(args.d,shape_input,round(m,3),round(s,3)))
    
    torch.cuda.empty_cache()
