#!/bin/sh

mkdir -p model_structures

# Cityscapes
python3 mem.py --d 18 --struct True > model_structures/resnet18_cityscapes.txt
python3 mem.py --d 34 --struct True > model_structures/resnet34_cityscapes.txt
python3 mem.py --d 50 --struct True > model_structures/resnet50_cityscapes.txt
python3 mem.py --d 101 --struct True > model_structures/resnet101_cityscapes.txt

#Pascal-VOC
python3 mem.py --i "pascal" --d 18 --struct True > model_structures/resnet18_pascal.txt
python3 mem.py --i "pascal" --d 34 --struct True > model_structures/resnet34_pascal.txt
python3 mem.py --i "pascal" --d 50 --struct True > model_structures/resnet50_pascal.txt
python3 mem.py --i "pascal" --d 101 --struct True > model_structures/resnet101_pascal.txt
