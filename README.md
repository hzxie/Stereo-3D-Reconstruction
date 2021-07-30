# Stereo 3D Reconstruction

This repository contains the source code for the paper [Toward 3D Object Reconstruction from Stereo Images](https://arxiv.org/abs/1910.08223.pdf).

**Important Note:** The source code is in the ([Stereo2Voxel](https://github.com/hzxie/Stereo-3D-Reconstruction/tree/Stereo2Voxel)/[Stereo2Point](https://github.com/hzxie/Stereo-3D-Reconstruction/tree/Stereo2Point)) branches of the repository.

![Overview](https://infinitescript.com/wordpress//wp-content/uploads/2021/02/Stereo-3D-Reconstruction-Overview.jpg)

## Cite this work

```
@article{xie2021towards,
  title={Toward 3D Object Reconstruction from Stereo Images},
  author={Xie, Haozhe and
          Tong, Xiaojun and
          Yao, Hongxun and
          Zhou, Shangchen and
          Zhang, Shengping and
          Sun, Wenxiu},
  journal={Neurocomputing},
  year={2021}
}
```

## Datasets

We use the [StereoShapeNet](https://www.shapenet.org/) dataset in our experiments, which is available below:

- [StereoShapeNet](https://gateway.infinitescript.com/?fileName=StereoShapeNet)

## Pretrained Models

The pretrained models on StereoShapeNet are available as follows:

- [Stereo2Voxel for StereoShapeNet](https://gateway.infinitescript.com/?fileName=Stereo2Voxel-StereoShapeNet.pth) (309 MB)
- [Stereo2Point for StereoShapeNet](https://gateway.infinitescript.com/?fileName=Stereo2Point-StereoShapeNet.pth) (356 MB)

## Prerequisites

#### Clone the Code Repository

```
git clone https://github.com/hzxie/Stereo-3D-Reconstruction.git
```

#### Install Python Denpendencies

```
cd Stereo-3D-Reconstruction
pip install -r requirements.txt
```

#### Train/Test Stereo2Voxel

```
git checkout Stereo2Voxel
```

#### Train/Test Stereo2Point

```
git checkout Stereo2Point

cd extensions/chamfer_dist
python setup.py install --user
```

#### Update Settings in `config.py`

You need to update the file path of the datasets:

```
__C.DATASETS.SHAPENET.LEFT_RENDERING_PATH   = '/path/to/ShapeNetStereoRendering/%s/%s/render_%02d_l.png'
__C.DATASETS.SHAPENET.RIGHT_RENDERING_PATH  = '/path/to/ShapeNetStereoRendering/%s/%s/render_%02d_r.png'
__C.DATASETS.SHAPENET.LEFT_DISP_PATH        = '/path/to/ShapeNetStereoRendering/%s/%s/disp_%02d_l.exr'
__C.DATASETS.SHAPENET.RIGHT_DISP_PATH       = '/path/to/ShapeNetStereoRendering/%s/%s/disp_%02d_r.exr'
__C.DATASETS.SHAPENET.VOLUME_PATH           = '/path/to/ShapeNetVox32/%s/%s.mat'
```

## Get Started

To train GRNet, you can simply use the following command:

```
python3 runner.py
```

To test GRNet, you can use the following command:

```
python3 runner.py --test --weights=/path/to/pretrained/model.pth
```

## License

This project is open sourced under MIT license.
