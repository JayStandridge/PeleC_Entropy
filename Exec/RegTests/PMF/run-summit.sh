#!/bin/bash -l

#BSUB -J pelec-hdf5-summit
#BSUB -o pelec-hdf5-summit.o%J
#BSUB -P CMB138
#BSUB -W 30
#BSUB -nnodes 2
#BSUB -N

cmd() {
  echo "+ $@"
  eval "$@"
}

cmd "module unload xl"
cmd "module load gcc/10.2.0"
cmd "module load cuda/11.4.2"
cmd "module load netlib-lapack/3.9.1"
cmd "jsrun -n 12 -r 6 -a 1 -c 1 -g 1 ./PeleC3d.gnu.MPI.CUDA.ex pmf-hdf5-zfp.inp"
