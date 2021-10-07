#!/bin/sh

scp ./*.py aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/experiments/realworld/scrnaseq
scp ./job.slurm* aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/experiments/realworld/scrnaseq

scp -r ../../../pcpca aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/
scp -r ../../../clvm aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/

echo "Done."