#!/bin/bash
#SBATCH --job-name=dgcnn_train
#SBATCH --cpus-per-task=120
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=dgcnn_semseg_s3dis_train_%j.out
#SBATCH --error=dgcnn_semseg_s3dis_train_%j.err

# Set the number of tasks and GPUs directly
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8

# Execute the job using Singularity
srun singularity exec --nv --bind /ceph:/ceph /ceph/project/ce-7-740/containers/dgcnn.sif \
	python main_semseg_s3dis.py \
	--exp_name=semseg_s3dis_6 --test_area=6 \
	--batch_size=64 \
	--test_batch_size=32 \
	--num_workers=48 \
	--num_workers_test=32 \
	--lr=0.002 \
	--epochs=110 \
	--momentum=0.85
