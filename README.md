# LucidDreaming

[ArXiv Link](https://arxiv.org/abs/2312.00588) | [Project Webpage](https://www.zhaoningwang.com/LucidDreaming/)

## Overview
This repository provides scripts for generating 3D shapes and scenes with the LucidDreaming pipeline. The code is based on threestudio and the original implementation can be found [here](https://github.com/threestudio-project/threestudio)

## Dataset
The main dataset is seperated into two subsets, complex and normal. The prompt used to generate each sample is in the filename of the text files. The location of the dataset is at:
```
objects/multi_gen
```


## Run Script
Use the run script to start a training or inference job:
```
bash scripts/multi_gen/run.sh
```
You can find the specific arguments in the top of the run script.
