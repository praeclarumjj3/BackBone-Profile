#!/bin/sh

#ResNet18
python3 resnet.py --d 18 > resnet18_cityscapes_basic.txt
python3 resnet.py --i "pascal" --d 18 > resnet18_pascal_basic.txt
python3 resnet.py --b "bt" --d 18 > resnet18_cityscapes_bottle.txt
python3 resnet.py --i "pascal" --b "bt" --d 18 > resnet18_pascal_bottle.txt