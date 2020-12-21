#!/bin/sh

#ResNet34
python3 resnet.py --d 34 > resnet34_cityscapes_basic.txt
python3 resnet.py --i "pascal" --d 34 > resnet34_pascal_basic.txt
python3 resnet.py --b "bt" --d 34 > resnet34_cityscapes_bottle.txt
python3 resnet.py --i "pascal" --b "bt" --d 34 > resnet34_pascal_bottle.txt