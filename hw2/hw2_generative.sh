#!/bin/sh

python3.6 logistic.py "${3}" "${4}" "${5}" "${6}" --n_iter 1000  --batch_size 20 --alpha 0.01 --eta 1 --valid_ratio 0.01 --model "pgm"

