#!/bin/sh

python3 resnet.py --d 18 > resnet18.txt
python3 resnet.py --d 34 > resnet34.txt
python3 resnet.py --d 50 > resnet50.txt