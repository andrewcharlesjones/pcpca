#!/bin/sh

scp ./*.py aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/experiments/realworld/mouse_protein_expression
scp ./job.slurm* aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/experiments/realworld/mouse_protein_expression

scp -r ../../../pcpca aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/
scp -r ../../../clvm aj13@della.princeton.edu:/scratch/gpfs/aj13/pcpca/

echo "Done."