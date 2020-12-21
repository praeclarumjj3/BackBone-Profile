#!/bin/sh

python3 timer.py --d 18 > model_structures/resnet18.txt
python3 timer.py --d 34 > model_structures/resnet34.txt
python3 timer.py --d 50 > model_structures/resnet50.txt