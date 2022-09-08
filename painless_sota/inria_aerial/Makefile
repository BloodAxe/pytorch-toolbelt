NUM_GPUS=`python -c "import torch;print(torch.cuda.device_count())"`

tensorboard-daemon:
	nohup tensorboard --logdir=./runs --host 0.0.0.0 --port 7007 --window_title "Inria Aerial" > /dev/null &

env:
	conda remove --name xview3 --all;
	conda env create -f environment.yml

opencv:
	pip uninstall opencv-python;
	pip uninstall opencv-python-headless;
	pip install opencv-python-headless;

pull:
	git	pull



seg_r200d_unet: pull
	export OMP_NUM_THREADS=4
	INRIA_AERIAL_DATA_DIR=/home/eugene/inria_dataset torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
	model=seg_r200d_unet \
	dataset.train.loader.batch_size=16 \
	dataset.train.dataset.num_samples=16536 \
	dataset.validation.loader.batch_size=16 \
	runner.show_batches=False

seg_r200d_unet_rdtsc: pull
	export OMP_NUM_THREADS=4
	INRIA_AERIAL_DATA_DIR=/home/eugene/inria_dataset torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
	model=seg_r200d_unet_rdtsc \
	dataset.train.loader.batch_size=16 \
	dataset.train.dataset.num_samples=16536 \
	dataset.validation.loader.batch_size=16 \
	runner.show_batches=False

seg_r200d_unet_rdtsc_s2: pull
	export OMP_NUM_THREADS=4
	INRIA_AERIAL_DATA_DIR=/home/eugene/inria_dataset torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
	model=seg_r200d_unet_rdtsc_s2 \
	dataset.train.loader.batch_size=20 \
	dataset.train.dataset.num_samples=16536 \
	dataset.validation.loader.batch_size=16 \
	runner.show_batches=False

b6_unet32_s2_rdtc: pull
	export OMP_NUM_THREADS=4
	INRIA_AERIAL_DATA_DIR=/home/eugene/inria_dataset torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
	model=b6_unet32_s2_rdtc \
	dataset.train.loader.batch_size=16 \
	dataset.train.dataset.num_samples=4096 \
	dataset.validation.loader.batch_size=16 \
	runner.show_batches=False

seg_repvgg_unet: pull
	export OMP_NUM_THREADS=4
	INRIA_AERIAL_DATA_DIR=/home/eugene/inria_dataset torchrun --standalone --nnodes=1 --nproc_per_node=4 train.py \
	model=seg_repvgg_unet \
	optimizer=adamw_scratch \
	dataset.train.loader.batch_size=32 \
	dataset.train.dataset.num_samples=16536 \
	dataset.validation.loader.batch_size=16 \
	runner.show_batches=False