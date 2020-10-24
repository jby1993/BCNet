# BCNet

This repository includes the some source code and related dataset of paper BCNet: Learning Body and Cloth Shape from A Single Image, ECCV 2020, [https://arxiv.org/abs/2004.00214](https://arxiv.org/abs/2004.00214).

Authors: Boyi Jiang, [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/), Yang Hong, Jinhao Luo, [Ligang Liu](http://staff.ustc.edu.cn/~lgliu/), and [Hujun Bao](http://www.cad.zju.edu.cn/bao/).

Note that all of the code and dataset can be only used for research purposes. If you are interested in business purposes/for-profit use, please contact Juyong Zhang (the corresponding author, email: [juyong@ustc.edu.cn](mailto:juyong@ustc.edu.cn)).

## Dataset
### 1. Synthetic Dataset
- Download and visualization the Synthetic Dataset follow the instruction in  body_garment_dataset.
### 2. HD Texture Dataset
- For now, we are unable to release the full training data due to the restriction of commertial scans.

## Inference
### 1. Install
- Download the [trained model](https://mailustceducn-my.sharepoint.com/:u:/g/personal/jby1993_mail_ustc_edu_cn/Ec4gw5LLQqRFqHnik_JU-mQBja5BiH7uWZJXTyQoCJsqMg?e=azQfQL) and mv to the "models" folder.
- Download the [tmps](https://mailustceducn-my.sharepoint.com/:f:/g/personal/jby1993_mail_ustc_edu_cn/EmZqxx-OSo9FiaUmNTP-sisBBlpUM0W4CLOkSJwBcrGogA?e=fgubX0) data in body_garment_dataset.
- This code is compatible with python 3.7 and pytorch 1.6. In addition, the following packages are required: numpy, torch_geometric 1.5, openmesh, opencv...
You can create an anaconda environment called BCNet with the required dependencies by running: 
```
conda env create -f environment.yml
conda activate BCNet
```
### 2. Usage
You can generate the results in recs for example images by running the code:
```
cd code
python infer.py --inputs ../images --save-root ../recs
```



## Citation

Please cite the following paper if it helps your research:
```
@inproceedings{jiang2020bcnet,
  title={BCNet: Learning Body and Cloth Shape from A Single Image},
  author={Jiang, Boyi and Zhang, Juyong and Hong, Yang and Luo, Jinhao and Liu, Ligang and Bao, Hujun},
  booktitle={European Conference on Computer Vision},
  year={2020},
  organization={Springer}
}
```