# ActCLR

This is an official PyTorch implementation of **Actionlet-Dependent Contrastive Learning for Unsupervised Skeleton-Based Action Recognition" in CVPR 2023**. 

## Requirements
  ![Python >=3.6](https://img.shields.io/badge/Python->=3.6-yellow.svg)    ![PyTorch >=1.6](https://img.shields.io/badge/PyTorch->=1.4-blue.svg)

## Data Preparation
- Download the raw data of [NTU RGB+D](https://github.com/shahroudy/NTURGB-D) and [PKU-MMD](https://www.icst.pku.edu.cn/struct/Projects/PKUMMD.html).
- For NTU RGB+D dataset, preprocess data with `tools/ntu_gendata.py`. For PKU-MMD dataset, preprocess data with `tools/pku_part1_gendata.py`.
- Then downsample the data to 50 frames with `feeder/preprocess_ntu.py` and `feeder/preprocess_pku.py`.
- If you don't want to process the original data, download the file folder in Google Drive [action_dataset](https://drive.google.com/drive/folders/1VnD3CLcD7bT5fMGI3tDGPlcWZmBbXS0m?usp=sharing) or BaiduYun link [action_dataset](https://pan.baidu.com/s/1NRK1ksRHgng_NkOO1ZYTcQ), code: 0211. NTU-120 is also provided: [NTU-120-frame50](https://drive.google.com/drive/folders/1dn8VMcT9BYi0KHBkVVPFpiGlaTn2GnaX?usp=sharing).

## Installation
  ```bash
  
# Install other python libraries
$ pip install -r requirements.txt
  ```

## Unsupervised Pre-Training

This work is a two-phase training, the first phase is trained using the AimCLR method, you can refer to the code of [AimCLR](https://github.com/Levigty/AimCLR).

The second phase of training builds on the pre-trained parameters of the first phase. Example for unsupervised pre-training of **3s-ActCLR**. You can change some settings of `.yaml` files in `config/ntu60/pretext` folder.
```bash
# train on NTU RGB+D xview joint stream
$ CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 main.py pretrain_actclr --config ./config/ntu60/pretext/pretext_actclr_xview_joint.yaml
```

## Linear Evaluation

The linear evaluation process can be executed by running the script located in the `AimCLR` directory.
Example for linear evaluation of **3s-ActCLR**. You can change `.yaml` files in `config/ntu60/linear_eval` folder.
```bash
# Linear_eval on NTU RGB+D xview
$ python main.py linear_evaluation --config config/ntu60/linear_eval/linear_eval_actclr_xview_joint.yaml
```

## Results

|     Model     | NTU 60 xsub (%) | NTU 60 xview (%) |
| :-----------: | :-------------: | :--------------: |
| ActCLR-joint  |      80.9      |      86.7       |
| ActCLR-motion |      78.6      |      84.4       |
|  ActCLR-bone  |      80.1      |      85.0       |
|   3s-ActCLR   |    **84.3**    |    **88.8**     |


## Citation
Please cite our paper if you find this repository useful in your resesarch:

```
@inproceedings{lin2023actionlet,
  Title= {Actionlet-Dependent Contrastive Learning for Unsupervised Skeleton-Based Action Recognition},
  Author= {Lin, Lilang and Zhang, Jiahang and Liu, Jiaying},
  Booktitle= {CVPR},
  Year= {2023}
}
```

## Acknowledgement
The framework of our code is extended from the following repositories. We sincerely thank the authors for releasing the codes.
- The framework of our code is based on [AimCLR](https://github.com/Levigty/AimCLR).
- The encoder is based on [ST-GCN](https://github.com/yysijie/st-gcn/blob/master/OLD_README.md).

## Licence

This project is licensed under the terms of the MIT license.