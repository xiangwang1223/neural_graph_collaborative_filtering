# Neural Graph Collaborative Filtering
This is our Tensorflow implementation for the paper:

>Xiang Wang, Xiangnan He, Meng Wang, Fuli Feng, and Tat-Seng Chua (2019). [Neural Graph Collaborative Filtering](https://www.comp.nus.edu.sg/~xiangnan/papers/sigir19-NGCF.pdf). In SIGIR'19, Paris, France, July 21-25, 2019.

Author: Dr. Xiang Wang (xiangwang at u.nus.edu)

## Introduction
Neural Graph Collaborative Filtering (NGCF) is a new recommendation framework based on graph neural network, explicitly encoding the collaborative signal in the form of high-order connectivities in user-item bipartite graph by performing embedding propagation.

## Citation 
If you want to use our codes in your research, please cite:
```
@inproceedings{NGCF19,
  author    = {Xiang Wang and
               Xiangnan He and
               Meng Wang and
               Fuli Feng and
               Tat-Seng Chua},
  title     = {Neural Graph Collaborative Filtering},
  booktitle = {{SIGIR}},
  year      = {2019}
}
```
## Environment Requirement
The code has been tested running under Python 3.6.5. The required packages are as follows:
* tensorflow == 1.8.0
* numpy == 1.14.3
* scipy == 1.1.0
* sklearn == 0.19.1

## Example to Run the Codes
