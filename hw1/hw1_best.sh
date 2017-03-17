#!/bin/sh

python3.6 linear_regression.py ${1} ${2} ${3} --l 0.2 --n_prev 9 --train_ratio 0.8
