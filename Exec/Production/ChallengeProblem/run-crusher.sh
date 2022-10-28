#!/bin/bash -l

#SBATCH -J pelec-challenge-crusher
#SBATCH -o %x.o%j
#SBATCH -A CMB138_crusher
#SBATCH -t 0:30:00
#SBATCH -N 4

set -e

cmd() {
  echo "+ $@"
  eval "$@"
}

cmd "module unload PrgEnv-cray"
cmd "module load PrgEnv-amd"
cmd "module load rocm/5.1.0"
cmd "srun -N4 -n32 -c1 --gpus-per-node=8 --gpu-bind=closest ./PeleC3d.hip.x86-trento.TPROF.MPI.HIP.ex challenge.inp max_step=10 amr.max_level=1 amr.plot_int=5 ascent.plot_int=2"
