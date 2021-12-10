#!/usr/bin/env bash

ncpu=$(nproc --all)
echo "Number of Cores: $ncpu"

mpirun python test.py 10 &
mpirun python test.py 2 &

echo "Both scripts started"
wait
echo "All done!"
