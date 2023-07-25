# Transcendental Idealism of Planner:<br>Evaluating Perception from the Planning Perspective for Autonomous Driving

*International Conference on Machine Learning* ([ICML 2023](https://icml.cc/Conferences/2023))<br>

[Wei-Xin Li](http://www.svcl.ucsd.edu/~nicolas/), [Xiaodong Yang](https://xiaodongyang.org/) <br>
[[Paper]](https://arxiv.org/pdf/2306.07276.pdf) [[Poster]](poster.pdf)

**Abstract:**
*Evaluating the performance of perception modules in autonomous driving is
one of the most critical tasks in developing the complex intelligent
system. While module-level unit test methodologies adopted from
traditional computer vision tasks are still feasible to a certain extent,
it remains far less explored to evaluate the impact of perceptual noise on
the driving performance of autonomous vehicles in a consistent and
holistic manner.
In this work, we propose a principled framework that
provides a coherent and systematic understanding of how an error in the
perception module affects the planning of an autonomous vehicle that
actually controls the vehicle. Specifically, the planning process is
formulated as expected utility maximisation, where all input signals from
upstream modules jointly provide a world state description, and the
planner strives for the optimal action to execute by maximising the
expected utility determined by both world states and actions. We show
that, under practical conditions, the objective function can be
represented as an inner product between the world state description and
the utility function in a Hilbert space. This geometric interpretation
enables a novel way to analyse the impact of noise in world state
estimation on planning and leads to a universal
quantitative metric for the purposes. The whole framework resembles the
idea of transcendental idealism in classical philosophical literature,
which gives the name to our approach.*

This repository contains an example implementeation for a neural planner to exemplify our work.

**Note:**
- The dynamic range of the TIP score depends on that of the utility function. In this example implementation, the probability density map on a discrete grid is used in lieu of the utility function for the neural planner, thus the TIP score ranges between [-2, 0]. More context is available in the paper.
- This repository is built on the [nuscenes-devkit library](https://github.com/nutonomy/nuscenes-devkit/blob/master/LICENSE.txt), and is adapted from the implementation of [Planner-KL Divergence (PKL)](https://github.com/nv-tlabs/planning-centric-metrics/blob/master/LICENSE), both of which use the Apache 2.0 License.

### Citation
If you found this codebase useful in your research or work, please consider citing
```
@InProceedings{Li_2023_ICML,
author = {Li, Wei-Xin and Yang, Xiaodong},
title = {Transcendental Idealism of Planner:
Evaluating Perception from the Planning Perspective for Autonomous Driving},
booktitle = {Proceedings of the 40th International Conference on Machine Learning (ICML)},
month = {July},
year = {2023}
}
```

## Preparation
Download nuscenes data and maps from [https://www.nuscenes.org/](https://www.nuscenes.org/). We assume that [nuScenes](https://www.nuscenes.org/download) is located at `NUSCENES_ROOT` and nuScenes maps are located at `NUSCENES_MAP_ROOT`.

Install dependencies.

```
pip install nuscenes-devkit tensorboardY planning-centric-metrics
```

If neccesary, follow the [full tutorial](https://github.com/nv-tlabs/planning-centric-metrics#full-tutorial) of PKL to visulise the data, and train the neural planner (including the masks file).

Install this `ti-planner` package.
```
pip install -e .
```

## Score A Submission With TIP
Score a nuScenes detection submission on an eval set. Set `--gpuid=-1` to use the CPU.
```
python main.py eval_test VERSION EVAL_SET --result_path=SUBMISSION --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT --model_path=MODEL_PATH --mask_json=MASK_PATH --output_dir=OUTPUT_DIR
```
For instance, one can score a synthetic submission where all detections are removed from the ground truth by running
```
python main.py eval_test mini mini_train --result_path='./example_submission.json' --nworkers=8 --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT --plot_kextremes=5 --output_dir='./result' --model_path='./planner.pt' --mask_json='./masks_trainval.json'
```
with output results
```
./result/planner_metric_tip.json
./result/TIP_worst0000.png
./result/TIP_worst0001.png
./result/TIP_worst0002.png
./result/TIP_worst0003.png
./result/TIP_worst0004.png
```
and statistics printout
```
{'min': -0.16451147198677063, 'max': -0.00033551082015037537, 'mean': -0.051448892802000046, 'median': -0.02976861782371998, 'std': 0.052457597106695175, 'top_worst': {311: -0.16451147198677063, 316: -0.16401930153369904, 317: -0.16395069658756256, 319: -0.16393756866455078, 322: -0.16389690339565277}}
```
5 scenes with the lowest TIP scores under the trained planner are (the lower the TIP score is, the worse the error is)
<img src="./imgs/tip_worst.gif">

## TIP Score Distribution
Given the TIP scores produced from [MEGVII val detections](https://github.com/poodarchu/Det3D/tree/master/examples/cbgs) by the scoring functionality above, one can visualise the distribution of TIP scores by
```
python main.py eval_test trainval val --result_path=SCORE_JSON_PATH --nworkers=8 --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT
```
```
python main.py score_distribution_plot trainval --score_result_path=SCORE_JSON_PATH --dataroot=NUSCENES_ROOT
```
<img src="./imgs/dist_tip.png">

## Sensitivity to False Negatives and False Positives by TIP
To understand the impact of false negatives, one can visualise the
significance of each ground-truth object by removing it from the correponding scene and computing the TIP.
```
python main.py false_neg_viz trainval --model_path=MODELPATH --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT
```
<img src="./imgs/fneg_tip.gif">

Similarly, one can visualise the importance of correctly not predicting a false positive at each location in a grid around the ego.
```
python main.py false_pos_viz trainval --model_path=MODELPATH --dataroot=NUSCENES_ROOT --map_folder=NUSCENES_MAP_ROOT
```
<img src="./imgs/fpos_tip.gif">

## Acknowledgements
The work is supported by the perception, planning, infrastructure, and
machine learning teams at [QCraft Inc.](https://www.qcraft.ai/en), with our
special acknowledgements to
Tiancheng Chen, Haijun Yang, Hao Lei, Renjie Li, Boqian Yu, Xiang Wang,
and Chenxu Luo for informative discussion and timely assistance.
We also thank all anonymous reviewers for their constructive feedbacks and
valuable suggestions to improve the work.
