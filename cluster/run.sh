#!/bin/bash
source /home/atatjer/src/scalinglawsquantization/syncuv.sh

# Job specific vars
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
python /home/atatjer/src/TrainingDynamicsQuantizationRobustness/gptqmodel_quantize.py  --config=$config --job_idx=$job_idx