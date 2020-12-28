#!/bin/sh

mkdir -p model_structures

# Cityscapes
python3 mem.py --struct True > model_structures/xception_cityscapes.txt

#Pascal-VOC
python3 mem.py --i "pascal" --struct True > model_structures/xception_pascal.txt
