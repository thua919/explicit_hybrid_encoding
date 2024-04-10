
## Explicit Hybrid Encoding

This repository has been developed largely based on [H2-Mapping](https://github.com/SYSU-STAR/H2-Mapping). We express our sincere gratitude to the authors for their wonderful work! It should be noted that our Explicit Hybrid Encoding is rooted in `src/functions/single_grid_net.py` and is initiated by the `decoder: single_grid_net` configuration in `configs/scannet`.



Plase begin by cloning this repository and all its submodules using the following command:

```bash
git clone https://github.com/thua919/explicit_hybrid_encoding
```
and then, we kindly direct developers to the original [H2-Mapping repository](https://github.com/SYSU-STAR/H2-Mapping) for installation instructions. Once done, please follow the data downloading procedure on [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

<details>
  <summary>[Directory structure of ScanNet (click to expand)]</summary>
  
  DATAROOT is `Datasets` by default. If a sequence (`sceneXXXX_XX`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── scannet
      └── scans
          └── scene0000_00
              └── frames
                  ├── color
                  │   ├── 0.jpg
                  │   ├── 1.jpg
                  │   ├── ...
                  │   └── ...
                  ├── depth
                  │   ├── 0.png
                  │   ├── 1.png
                  │   ├── ...
                  │   └── ...
                  ├── intrinsic
                  └── pose
                      ├── 0.txt
                      ├── 1.txt
                      ├── ...
                      └── ...

```
</details>

Once the data is downloaded and set up properly, please quickly test the running it by:

```bash
python -W ignore run_mapping.py configs/ScanNet/scene0000.yaml
```




## Citing

Please consider citing following works when you use this repository:

```BibTeX
@ARTICLE{10243098,
  author={Jiang, Chenxing and Zhang, Hanwen and Liu, Peize and Yu, Zehuan and Cheng, Hui and Zhou, Boyu and Shen, Shaojie},
  journal={IEEE Robotics and Automation Letters}, 
  title={H$_{2}$-Mapping: Real-Time Dense Mapping Using Hierarchical Hybrid Representation}, 
  year={2023},
  volume={8},
  number={10},
  pages={6787-6794},
  doi={10.1109/LRA.2023.3313051}}

@inproceedings{nerfslam24hua,
  author = {Johari, M. M. and Carta, C. and Fleuret, F.},
  title = {Benchmarking Implicit Neural Representation and Geometric Rendering in Real-Time RGB-D SLAM},
  booktitle = {Proceedings of the IEEE international conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2023},
}
```
