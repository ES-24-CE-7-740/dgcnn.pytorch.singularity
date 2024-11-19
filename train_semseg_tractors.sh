#!/bin/bash
#SBATCH --job-name=dgcnn_train
#SBATCH --cpus-per-task=120
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --output=dgcnn_tractors_train_%j.out
#SBATCH --error=dgcnn_tractors_train_%j.err

# Set the number of tasks and GPUs directly
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8

# Execute the job using Singularity
srun singularity exec --nv --bind /ceph:/ceph /ceph/project/ce-7-740/containers/dgcnn.sif \
	python main_semseg_tractors.py \
	--exp_name=semseg_tractors_sgd --test_area=6 \
	--batch_size=16 \
	--batch_size_test=8 \
	--epochs=150 \
	--use_adam \
	--lr=0.0005 \
	--momentum=0.9 \
	--scheduler=cos \
	--num_points=100000 \
	--dropout=0.5 \
	--num_workers=64 \
	--num_workers_test=32 \
	--num_classes=3 \
	--n_features=9 \
	--data_root=/ceph/project/ce-7-740/data/tractors_and_combines_synth/

# --use_adam
# --model_root <pretrained_model_path>
