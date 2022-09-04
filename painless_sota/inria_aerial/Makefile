NUM_GPUS=`python -c "import torch;print(torch.cuda.device_count())"`

tensorboard-daemon:
	nohup tensorboard --logdir=./runs --host 0.0.0.0 --port 6006 --window_title "EOS Detection" > tb.log &

env:
	conda remove --name xview3 --all;
	conda env create -f environment.yml

opencv:
	pip uninstall opencv-python;
	pip uninstall opencv-python-headless;
	pip install opencv-python-headless;

pull:
	git	pull


grid_search: pull
	torchrun --standalone --nnodes=1 --nproc_per_node=$(NUM_GPUS) train.py -m \
	model=seg_repvgg_unet,b6_unet32_s2_rdtc,seg_r200d_unet \
	batch_size=16
