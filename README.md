# 2D Floor Plan to 3D model
Python framework of several existing methods and datasets for image segmentation.
This repository origin from work Gijs de Jong: [repository](https://github.com/TheOnlyError/2d3d)

Thesis: **[Multi-Unit Floor Plan Recognition and Reconstruction](https://repository.tudelft.nl/islandora/object/uuid%3A158f6745-0b43-4796-b21d-6388a35f5a2d?collection=education)**
<br>
[Gijs de Jong](https://github.com/TheOnlyError)
<br>

Contents:
* [Creation of virtual environment](#creation-of-virtual-environment)
* [Data preparation](#data-preparation)
* [Training](#training)
* [Testing](#testing)
* [Evaluation](#evaluation)

<details>
<summary>Supported models and datasets</summary>

Models:
- [CubiCasa5k](https://arxiv.org/pdf/1904.01920.pdf)
- [DFPR](https://arxiv.org/pdf/1908.11025.pdf)
- [Unet](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28)
- [Unet++](https://arxiv.org/pdf/1912.05074.pdf)
- [Unet3+](https://www.semanticscholar.org/paper/UNet-3%2B%3A-A-Full-Scale-Connected-UNet-for-Medical-Huang-Lin/0b444f74dd9cc06c2833dd15f9258ef5e169e6ea)
- Our models CAB1, CAB2

Dataset:
- [R3D](https://www.cs.toronto.edu/~fidler/projects/rent3D.html)
- [CubiCasa5k](https://zenodo.org/records/2613548)
- [MURF](resources)
- [CVC-FP](http://dag.cvc.uab.es/resources/floorplans/)
- [MLSTRUCT-FP](https://github.com/MLSTRUCT/MLSTRUCT-FP)


</details>

## Creation of virtual environment
To create and setup venv install `python3.8` and `virtualenv`:

For Linux users:
```bash
cd <project_folder>
python3.8 -m venv venv  # Create virtual environment
source venv/bin/activate  # Virtual environment activation 
pip install --upgrade pip
pip install -r requirements.txt  # Or you could try 'specific_requirements.txt' to get identical environment
```
For Windows users:
```bash
cd <project_folder>
python3.8 -m venv venv  # Create virtual environment
source venv/Scripts/activate.bat  # Virtual environment activation 
pip install --upgrade pip
pip install -r requirements.txt  # Or you could try 'specific_requirements.txt' to get identical environment
```
## Data preparation
For data preparation follow example of dataset creation:

[Dataset example](data_create_example.py)

Framework use `tfrecord` for training and have some tools for its creation: 
> (Here is example for **R3D**)
> - [split](datasets/split_dataset.py) - generate data description from folder
> - [create_mask](datasets/create_data_mask.py) - generate binary mask for each class
> - [create_dataset](datasets/floorplans.py) - generate tfrecords
> - [Config](utils/config.py) - generate Config file used for training
> - (optional) [create_overlay](datasets/create_data_mask.py) - visualize mask on original data
> - (optional) [augmentation](datasets/augment.py) - create more data with rotation
> - (optional) [create_heatmap_from_paths](datasets/create_heatmap.py) - generate heatmaps for training

## Training
For training, you will need data in `tfrecord` file and Config with model parameters. For generation one could follow
[train_example](configs/train_example.py).

When you have prepared data and config, you can run training by this command:
> Make sure you had installed [virtual environment](#creation-of-virtual-environment) and it is active
```bash
python train_config.py <config> 
```
You can add more configs to run, just by adding them and add some parameters from CLI to update your config. 
But be careful, if you add more configs, CLI parameters overwrite parameters in all passed configs. All parameters could be seen by using `-h`.

## Testing
```bash
python predict.py
```
## Evaluation
```bash
python evaluate.py
```
Or 
```bash
python evaluate_config.py <config>
```
## TODO:
- [x] Add Config environment - Done from [mmcv](https://github.com/open-mmlab/mmcv/tree/v1.7.1)
- [ ] Add [InternImage](https://github.com/OpenGVLab/InternImage/tree/master) backbone
- [ ] Add [FloorPlanCAD](https://floorplancad.github.io/) dataset, [model](https://arxiv.org/pdf/2105.07147v2.pdf)
- [x] Add [MLSTRUCT-FP](https://github.com/MLSTRUCT/MLSTRUCT-FP)
- [x] Add [CVC-FP](http://dag.cvc.uab.es/resources/floorplans/)

# License
[Apache License v2.0](LICENSE)
