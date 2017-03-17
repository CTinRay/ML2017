#!/bin/sh

python3.6 baseline.py ${1} ${2} ${3}  --valid_ratio 0.2 --l 0.2 --n_prev 9 --rate 0.0001 --stop 0.1 --train_ratio 1 
