# Deep Anomaly Detection Using Geometric Transformations
To be presented in NIPS 2018 by Izhak Golan and Ran El-Yaniv.

## Introduction
This is the official implementation of "Deep Anomaly Detection Using Geometric Transformations".
It includes all experiments reported in the paper.

## Requirements
* Python 3.5+
* Keras 2.2.0
* Tensorflow 1.8.0
* sklearn 0.19.1
* Pillow 7.0.0

## How to start
Run experiments.py first to get the results 
Then run showRes to show the roc curve 

## ROC of sliding window 
* Mar. 28th
We firstly come up with a naive sliding window method, and we have our results like below:

![](img/slide_roc.png)


## Citation
If you use the ideas or method presented in the paper, please cite:

```
@article{golan2018deep,
  title={Deep Anomaly Detection Using Geometric Transformations},
  author={Golan, Izhak and El-Yaniv, Ran},
  journal={arXiv preprint arXiv:1805.10917},
  year={2018}
}
```
