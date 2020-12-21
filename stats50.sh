#!/bin/sh

#ResNet50
python3 resnet.py --d 50 > resnet50_cityscapes_basic.txt
python3 resnet.py --i "pascal" --d 50 > resnet50_pascal_basic.txt
python3 resnet.py --b "bt" --d 50 > resnet50_cityscapes_bottle.txt
python3 resnet.py --i "pascal" --b "bt" --d 50 > resnet50_pascal_bottle.txt