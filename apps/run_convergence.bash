#!/usr/bin/env bash

read -p "Alpha: " alpha

export PYTHONPATH=.

for ((i = 2 ; i <= 6; i++)); do
    echo "Running n =" $i
    mpirun -np 1 python apps/convergence.py $alpha $i &
done

wait
echo "All done!"