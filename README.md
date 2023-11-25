# HRTF-SCNN
Official implementation of the paper "Head-Related Transfer Function Interpolation with a Spherical CNN".
[[arXiv](https://arxiv.org/abs/2307.14013)]

##  Note
Our paper has been submitted to ICASSP 2024, and we intend to release the complete code after the results are announced on December 13th. Stay tuned for updates!

Here's a brief overview of the repository's structure:

- **dataset.py**: [Coming Soon] The `dataset.py` file will provide the necessary data preprocessing and loading functions. 

- **model.py**: This file contains the implementation of the learning model used in this project. It includes layers and the model architecture.

- **train.py**: [Coming Soon] The `train.py` file will contain the code for training the model using the HUTUBS dataset. 


## Citation
```
@article{chen2023head,
  title={Head-Related Transfer Function Interpolation with a Spherical CNN},
  author={Chen, Xingyu and Ma, Fei and Zhang, Yile and Bastine, Amy and Samarasinghe, Prasanga N},
  journal={arXiv preprint arXiv:2309.08290},
  year={2023}
}
```
## Reference
```
Cohen, T. S., Geiger, M., Köhler, J., & Welling, M. (2018). Spherical cnns. arXiv preprint arXiv:1801.10130.

Esteves, C., Allen-Blanchette, C., Makadia, A., & Daniilidis, K. (2018). Learning so (3) equivariant representations with spherical cnns. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 52-68).

Brinkmann, F., Dinakaran, M., Pelzer, R., Wohlgemuth, J. J., Seipel, F., Voss, D., ... & Weinzierl, S. (2019). The HUTUBS head-related transfer function (HRTF) database.
```
