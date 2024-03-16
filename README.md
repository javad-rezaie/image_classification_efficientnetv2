# Image Classification with EfficientNetV2

This repository contains code for training an image classification model using EfficientNetV2 architecture. It includes instructions for setting up the environment, downloading the dataset, and running the training process.

## Setup

1. Clone this repository:

   ```shell
   git clone git@github.com:javad-rezaie/image_classification_efficientnetv2.git
   ```

2. Create a Conda environment and activate it:

   ```shell
   conda create --name openmmlab python=3.10 -y
   conda activate openmmlab
   ```

3. Install the necessary packages:

   ```shell
   conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
   sudo reboot  # Reboot your system to ensure proper CUDA setup
   git clone https://github.com/open-mmlab/mmpretrain.git
   cd mmpretrain
   pip install -U openmim && mim install -e .
   ```

## Downloading Dataset

Download the Stanford Cars dataset from Kaggle. You can find the dataset [here](https://www.kaggle.com/jessicali9530/stanford-cars-dataset).

Change the `data_root` path in `efficientnetv2_b0_config.py` to the location where you downloaded the data:

```python
data_root = "/path/to/Datasets/Stanford_Cars_by_class_folder/car_data/car_data/"
```

## Running Training

To start training, use the following command in the terminal. Feed the configuration file (`efficientnetv2_b0_config.py`) to the main `mmengine` running script (`main_train_mmengine.py`).

```shell
torchrun --nnodes 1 --nproc_per_node=3 main_train_mmengine.py efficientnetv2_b0_config.py
```

Feel free to adjust the `--nnodes` and `--nproc_per_node` parameters according to your hardware setup.

# Model Conversion to OpenVINO and Hugging Face Integration
## Converting to OpenVINO
Our trained PyTorch model was converted to OpenVINO format using the Model Optimizer tool. This streamlined the deployment process for various hardware platforms.

## Hugging Face Upload
We shared the OpenVINO model on the Hugging Face Model Hub, making it easily accessible for developers ([here](https://huggingface.co/spaces/homai/Stanford-Cars)). This allows for straightforward integration into applications and fine-tuning on custom datasets.

## Running on Hugging Face
Instantiating the model from its unique identifier on Hugging Face enables easy execution and result visualization. Whether through the website interface or the API, running the model is intuitive and efficient.

## Disclaimer

This project is intended for educational purposes only. Any use of this project for real-world applications should be done with caution and proper consultation with relevant experts.

## License

This project is licensed under the This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.


---

For more details and documentation, refer to the original EfficientNetV2 paper and the OpenMMLab repository: [EfficientNetV2 Paper](https://arxiv.org/abs/2104.00298), [OpenMMLab GitHub](https://github.com/open-mmlab).
