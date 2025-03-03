# [Human Motion Prediction under Unexpected Perturbation](https://openaccess.thecvf.com/content/CVPR2024/papers/Yue_Human_Motion_Prediction_Under_Unexpected_Perturbation_CVPR_2024_paper.pdf)
![](https://github.com/realcrane/Human-Motion-Prediction-under-Unexpected-Perturbation/blob/main/images/model.png)

We investigate a new task in human motion prediction, which is predicting motions under unexpected physical perturbation potentially involving multiple people. Compared with existing research, this task involves predicting less controlled, unpremeditated and pure reactive motions in response to external impact and how such motions can propagate through people. It brings new challenges such as data scarcity and predicting complex interactions. To this end, we propose a new method capitalizing differentiable physics and deep neural networks, leading to an explicit Latent Differentiable Physics (LDP) model. Through experiments, we demonstrate that LDP has high data efficiency, outstanding prediction accuracy, strong generalizability and good explainability. Since there is no similar research, a comprehensive comparison with 11 adapted baselines from several relevant domains is conducted, showing LDP outperforming existing research both quantitatively and qualitatively, improving prediction accuracy by as much as 70%, and demonstrating significantly stronger generalization.

## Get Started
### Dependencies
Below is the key environment with the recommended version under which the code was developed:  
  
 Python 3.8; torch 2.0.0; numpy 1.22.3; scipy 1.7.3; Cuda 11.1  

### Training
The differentiable IPM can be trained by using the training script in single-person/multi-people. The CVAEs and Samplers in the skeleton restoration model can be trained by using their corresponding training scripts in single-person-motion/multi-people-motion. The code is being optimized for improved readability and usability.

### Authors  
Jiangbei Yue, Baiyi Li, Julien Pettré, Armin Seyfried and He Wang
Jiangbei Yue scjy@leeds.ac.uk  
He Wang, he_wang@@ucl.ac.uk, [Personal Site](http://drhewang.com/)  
Project Webpage: https://drhewang.com/pages/LDP.html

### Contact  
If you have any questions, please feel free to contact me: Jiangbei Yue (scjy@leeds.ac.uk)  

### Acknowledgement  
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 899739 [CrowdDNA](https://crowddna.eu/).  

### Citation (Bibtex)  
Please cite our paper if you find it useful:
```
@inproceedings{yue2024human,
  title={Human Motion Prediction under Unexpected Perturbation},
  author={Yue, Jiangbei and Li, Baiyi and Pettr{\'e}, Julien and Seyfried, Armin and Wang, He},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1501--1511},
  year={2024}
}
```
