
install-conda:
	conda create --name openmmlab python=3.10 -y; \

install-pytorch:
	conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia; \
	sudo reboot

install-mmpretrain:
	git clone https://github.com/open-mmlab/mmpretrain.git; \
	cd mmpretrain; \
	pip install -U openmim && mim install -e .

train:
	torchrun --nnodes 1 --nproc_per_node=3 main_train_mmengine.py efficientnetv2_b0_config.py
