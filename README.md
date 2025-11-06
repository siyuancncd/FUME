# FUME

<p align="left">
  <img src="https://img.shields.io/badge/license-MIT-brightgreen" alt="License">
    <img src="https://komarev.com/ghpvc/?username=siyuancncd&repo=FUME" alt="GitHub Views">
</p>

Siyuan Duan, Yuan Sun, Dezhong Peng, Zheng Liu, Xiaomin Song, and Peng Hu. "[Fuzzy Multimodal Learning for Trusted Cross-modal Retrieval](https://openaccess.thecvf.com/content/CVPR2025/papers/Duan_Fuzzy_Multimodal_Learning_for_Trusted_Cross-modal_Retrieval_CVPR_2025_paper.pdf)". (CVPR 2025, PyTorch Code)

- Supplementary material is available at [here](https://openaccess.thecvf.com/content/CVPR2025/supplemental/Duan_Fuzzy_Multimodal_Learning_CVPR_2025_supplemental.pdf).
- Poster is available at [here](https://github.com/siyuancncd/FUME/blob/main/FUME_poster.png).
- Report is available at [here](https://mp.weixin.qq.com/s/LIOjHJ1n7C5hXJcEH8_81A).

## Abstract

Cross-modal retrieval aims to match related samples across distinct modalities, facilitating the retrieval and discovery of heterogeneous information. Although existing methods show promising performance, most are deterministic models and are unable to capture the uncertainty inherent in the retrieval outputs, leading to potentially unreliable results. To address this issue, we propose a novel framework called FUzzy Multimodal lEarning (FUME), which is able to self-estimate epistemic uncertainty, thereby embracing trusted cross-modal retrieval. Specifically, our FUME leverages the Fuzzy Set Theory to view the outputs of the classification network as a set of membership degrees and quantify category credibility by incorporating both possibility and necessity measures. However, directly optimizing the category credibility could mislead the model by over-optimizing the necessity for unmatched categories. To overcome this challenge, we present a novel fuzzy multimodal learning strategy, which utilizes label information to guide necessity optimization in the right direction, thereby indirectly optimizing category credibility and achieving accurate decision uncertainty quantification. Furthermore, we design an uncertainty merging scheme that accounts for decision uncertainties, thus further refining uncertainty estimates and boosting the trustworthiness of retrieval results. Extensive experiments on five benchmark datasets demonstrate that FUME remarkably improves both retrieval performance and reliability, offering a prospective solution for cross-modal retrieval in high-stakes applications. 

## Motivation

Problem of incredible results in cross-modal retrieval:
<p align="center">
<img src="https://github.com/siyuancncd/FUME/blob/main/FUME_problem1.png" width="500" height="200">
</p>

Counter-intuitive problem of Evidential Deep Learning in uncertainty
estimation:
<p align="center">
<img src="https://github.com/siyuancncd/FUME/blob/main/FUME_problem2.png" width="500" height="200">
</p>

## Framework

<p align="center">
<img src="https://github.com/siyuancncd/FUME/blob/main/FUME_framework.png">
</p>

The overview of the proposed Fuzzy Multimodal learning (FUME) method. Firstly, modality-specific DNNs ($f^I(\cdot)$ and $f^T(\cdot)$) project image modality samples $\mathcal X^I$ and text modality samples $\mathcal X^T$ into a common space. Secondly, the representations ($\mathbf{z}^I$ and $\mathbf{z}^T$) in the common space map to membership degrees ($\mathbf{m}^I$ and $\mathbf{m}^T$). Finally, the membership degrees and labels ($\mathcal Y$) are inputted into the function $\phi^{tr}(\cdot, \cdot)$
    to determine the \rev{category credibility during training ($\mathbf{r}^I$ and $\mathbf{r}^T$). These category credibilities} are then used to compute loss $\mathcal L_{fml}$ and supervise the learning of the entire neural network. At the same time, Consistency learning is employed to eliminate the cross-modal discrepancy. During the optimization process, FUME will reduce the decision uncertainty of each modality, ultimately reducing the cross-modal uncertainty.

## Experiments

<p align="center">
<img src="https://github.com/siyuancncd/FUME/blob/main/FUME_results.png">
</p>

## Requirements

```
Python==3.9.0
torch==2.3.1
torchvision==0.18.1
numpy==1.26.4
scikit-learn==1.5.0
scipy==1.13.1
```

## Datasets


Following [DSCMR](https://github.com/penghu-cs/DSCMR), these datasets can be downloaded from the following URLs:

* Wikipedia: http://www.svcl.ucsd.edu/projects/crossmodal/
* NUS-WIDE: http://lms.comp.nus.edu.sg/research/NUS-WIDE.htm
* XMediaNet: http://www.icst.pku.edu.cn/mipl/XMedia/

For the reason of the license, please download the raw data and extract the features using the VGG19 and Sentence CNN. Please refer to our paper to get the details of extraction.
For text, the Sentence CNN code can be obtained from https://github.com/yoonkim/CNN_sentence.

You can also use our provided features: [Baidu Disk](https://pan.baidu.com/s/1B497czcviS5eFEdEWRg39w?pwd=1122)

## Train and test

```
python main.py
```

## Citation

If this codebase is useful for your work, please cite our papers:

```
@inproceedings{duan2025fuzzy,
  title={Fuzzy Multimodal Learning for Trusted Cross-modal Retrieval},
  author={Duan, Siyuan and Sun, Yuan and Peng, Dezhong and Liu, Zheng and Song, Xiaomin and Hu, Peng},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20747--20756},
  year={2025}
}
```

```
@inproceedings{duandeep,
  title={Deep Fuzzy Multi-view Learning for Reliable Classification},
  author={Duan, Siyuan and Sun, Yuan and Peng, Dezhong and Duan, Guiduo and Peng, Xi and Hu, Peng},
  booktitle={Forty-second International Conference on Machine Learning}
}
```

## Question?

If you have any questions, please email siyuanduancn AT gmail DOT com.

## Future Work

1. Extend FUME to Multi-modal classification. (Accepted by **ICML 2025**: [Deep Fuzzy Multi-view Learning for Reliable Classification](https://github.com/siyuancncd/FUML)).
2. Existing uncertainty estimation mechanisms lack calibration.
3. Extend FUME to cross-modal retrieval tasks that are closer to real scenarios, such as ReID and healthcare.

## Acknowledgement

The code is inspired by 

* [Deep supervised cross-modal retrieval](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhen_Deep_Supervised_Cross-Modal_Retrieval_CVPR_2019_paper.html)
* [Evidential deep learning](https://github.com/dougbrion/pytorch-classification-uncertainty)
