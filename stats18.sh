#!/bin/sh

#ResNet18
python3 timer.py --d 18 > resnet18_stats/resnet18_cityscapes_basic.txt
python3 timer.py --i "pascal" --d 18 > resnet18_stats/resnet18_pascal_basic.txt
python3 timer.py --b "bt" --d 18 > resnet18_stats/resnet18_cityscapes_bottle.txt
python3 timer.py --i "pascal" --b "bt" --d 18 > resnet18_stats/resnet18_pascal_bottle.txt