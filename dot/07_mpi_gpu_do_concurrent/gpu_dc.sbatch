#!/bin/bash
#SBATCH -J dot_gpu_dc
#SBATCH -A ferrante
#SBATCH -p ckpt-all
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -t 00:15:00
#SBATCH -o dot_gpu_dc_%j.out

cd "$SLURM_SUBMIT_DIR"
module purge
module load nvhpc/24.5
module load gcc/11.2.0
srun nvidia-smi -L

CC=cc75,cc80,cc86,cc89,cc90
RUN="mpirun --bind-to none -x LD_LIBRARY_PATH -x DOT_FORCE_AWARE -np 2"

echo "===== Build A: managed memory ====="
mpif90 -O2 -stdpar=gpu -gpu=$CC -Minfo=stdpar dot_hybrid_gpu_dc.F90 -o dc_managed || echo "BUILD A FAILED"

echo "===== Build B: separate memory + OpenACC data ====="
mpif90 -O2 -DUSE_ACC -stdpar=gpu -acc=gpu -gpu=mem:separate,$CC -Minfo=accel,stdpar \
       dot_hybrid_gpu_dc.F90 -o dc_separate
if [ $? -ne 0 ]; then
  echo "mem:separate rejected, retrying with nomanaged"
  mpif90 -O2 -DUSE_ACC -stdpar=gpu -acc=gpu -gpu=nomanaged,$CC -Minfo=accel,stdpar \
         dot_hybrid_gpu_dc.F90 -o dc_separate || echo "BUILD B FAILED"
fi

export NVCOMPILER_ACC_NOTIFY=1

echo "===== Run 1: managed, staged ====="
[ -x dc_managed ] && $RUN ./dc_managed staged

echo "===== Run 2: separate, staged ====="
[ -x dc_separate ] && $RUN ./dc_separate staged

echo "===== Run 3: separate, aware ====="
[ -x dc_separate ] && DOT_FORCE_AWARE=1 $RUN ./dc_separate aware
