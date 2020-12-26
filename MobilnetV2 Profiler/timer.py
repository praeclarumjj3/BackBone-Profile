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


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

mstart = torch.cuda.Event(enable_timing=True)
mend = torch.cuda.Event(enable_timing=True)

tstart = torch.cuda.Event(enable_timing=True)
tend = torch.cuda.Event(enable_timing=True)

layers = {"Conv2D": [["e"],[0]],
        "Blocks": [],
        "ConvLast": [["e"],[0]],
        "AvgPool": [["e"],[0]],
        "Classifier": [["e"],[0]],}

def add_time(key,value,i=None):
    if key == "Blocks":
        if len(layers[key][i][0]) is 1:
            layers[key][i][0].append(value[0])
        layers[key][i][1].append(value[1])
    else:
        if len(layers[key][0]) is 1:
            layers[key][0].append(value[0])
        layers[key][1].append(value[1])

def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNActivation(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            activation_layer(inplace=True)
        )


# necessary for backwards compatibility
ConvBNReLU = ConvBNActivation


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        self.conv1 = ConvBNReLU(3, input_channel, stride=2, norm_layer=norm_layer)
        # building inverted residual blocks
        self.blocks = []
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                self.blocks.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        self.conv_last = ConvBNReLU(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer)
        
        for i in range(len(self.blocks)):
            layers["Blocks"].append([["e"],[0]])
        
        self.blocks = nn.ModuleList(self.blocks)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        
        shape = list(x.shape)
        mstart.record()
        x = self.conv1(x)
        mend.record()
        torch.cuda.synchronize()
        add_time("Conv2D",("with kernel_size=3, stride=2 and input shape: {}".format(shape),mstart.elapsed_time(mend)))

        for i in range(len(self.blocks)):
            shape = list(x.shape)
            mstart.record()
            x = self.blocks[i](x)
            mend.record()
            torch.cuda.synchronize()
            add_time("Blocks",("at position {} with input shape: {}".format(i+1,shape),mstart.elapsed_time(mend)),i)
        
        shape = list(x.shape)
        mstart.record()
        x = self.conv_last(x)
        mend.record()
        torch.cuda.synchronize()
        add_time("ConvLast",("with kernel_size=1, stride=1 and input shape: {}".format(shape),mstart.elapsed_time(mend)))
        
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        shape = list(x.shape)
        mstart.record()
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1)).reshape(x.shape[0], -1)
        mend.record()
        torch.cuda.synchronize()
        add_time("AvgPool",("with input shape: {}".format(shape),mstart.elapsed_time(mend)))
        
        shape = list(x.shape)
        mstart.record()
        x = self.classifier(x)
        mend.record()
        torch.cuda.synchronize()
        add_time("Classifier",("with Dropout=0.2 and dim=1280 input shape: {}".format(shape),mstart.elapsed_time(mend)))
        
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def mobilenet_v2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
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

    model = mobilenet_v2(False).to(device)
    warm_model = models.mobilenet_v2().to(device)
    model_whole = models.mobilenet_v2().to(device)
    if not os.path.exists('mobilenetv2_stats'):
        os.makedirs('mobilenetv2_stats')
    
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
        ## Conv2D
        m = np.mean(layers["Conv2D"][1][1:])
        s = np.std(layers["Conv2D"][1][1:])
        print("Average Execution Time for Conv2D {} is {} ms with std = {} ms".format(layers["Conv2D"][0][1],round(m,3),round(s,3)))

        ## Blocks
        for i in range(len(layers["Blocks"])):
            m = np.mean(layers["Blocks"][i][1][1:])
            s = np.std(layers["Blocks"][i][1][1:])
            print("Average Execution Time for Inverted Residual Block {} is {} ms with std = {} ms".format(layers["Blocks"][i][0][1],round(m,3),round(s,3)))

        ## Conv2D
        m = np.mean(layers["ConvLast"][1][1:])
        s = np.std(layers["ConvLast"][1][1:])
        print("Average Execution Time for ConvLast {} is {} ms with std = {} ms".format(layers["ConvLast"][0][1],round(m,3),round(s,3)))

        ## AvgPool
        m = np.mean(layers["AvgPool"][1][1:])
        s = np.std(layers["AvgPool"][1][1:])
        print("Average Execution Time for AvgPool {} is {} ms with std = {} ms".format(layers["AvgPool"][0][1],round(m,3),round(s,3)))

        ## Classifier
        m = np.mean(layers["Classifier"][1][1:])
        s = np.std(layers["Classifier"][1][1:])
        print("Average Execution Time for Classifier {} is {} ms with std = {} ms\n".format(layers["Classifier"][0][1],round(m,3),round(s,3)))

    ## Total
    m = np.sum(times)/iter
    s = np.std(times)
    print("Average Execution Time for MobileNetV2 with input shape: {} is {} ms with std = {} ms".format(shape_input,round(m,3),round(s,3)))
    
    torch.cuda.empty_cache()
