#!/bin/bash

python3.5 train_dnn.py ${1} /dev/null --n_iter 1000 --batch_size 128 --eta 0.001 --valid_ratio 0.1
