# [Recovery Analysis for Plug-and-Play Priors using the Restricted Eigenvalue Condition](https://arxiv.org/abs/2106.03668)

The plug-and-play priors (PnP) and regularization by denoising (RED) methods have become widely used for solving inverse problems by leveraging pre-trained deep denoisers as image priors.  While the empirical imaging performance and the theoretical convergence properties of these algorithms have been widely investigated, their recovery properties have not previously been theoretically analyzed.  We address this gap by showing how to establish theoretical recovery guarantees for PnP/RED by assuming that the solution of these methods lies near the fixed-points of a deep neural network. We also present numerical results comparing the recovery performance of PnP/RED in compressive sensing against that of recent compressive sensing algorithms based on generative models. Our numerical results suggest that PnP with a pre-trained artifact removal network provides significantly better results compared to the existing state-of-the-art methods.

Authored by: Jiaming Liu, M. Salman Asif, Brendt Wohlberg, and Ulugbek S. Kamilov

## How to run the code

### Prerequisites for PnP/RED
tqdm
python 3.6
scipy 1.2.1 or lower
numpy v1.17 or lower
pytorch 1.14 or lower

It is better to use Conda for installation of all dependecies.

### Models

The pre-trained models can be downloaded from [Google drive](https://drive.google.com/drive/folders/1qqBGX0-cI-2Jck3tGnT1OjCkIuTTS-To?usp=sharing). Once downloaded, place them into `./model_zoo`. Note that the shared `.pth/.pkl` file may contains parameters other than models themselves, such as optimizers and losses.

|Model|# Layers|# params|
|---|:--:|:---:|
|[color-AR](https://drive.google.com/drive/folders/1Irbacw8JEUuCOXqgk9Nuq555kcsYkpNI?usp=sharing)     | 12 | 3.20M |
|[color-Denoising](https://drive.google.com/drive/folders/16LuRaW6SBqUOD7EFiKsnWXhQcxZokM-Y?usp=sharing)     | 12 | 9.46M |
|[mri-AR](https://drive.google.com/drive/folders/1iq9i7wUJqkvsKXH8Z8D3wjWEZEcEoNeB?usp=sharing)| 12 | 3.18M  |
|[mri-Denoising](https://drive.google.com/drive/folders/1z8UFuauD0qnpKggxOunjTu_yA03SGgzc?usp=sharing)| 12 | 14.10M  |
|[gray-AR](https://drive.google.com/drive/folders/1vBvsHfzcCLkwXBTDff49L4H3pcQR3Qa7?usp=sharing)| 12 | 9.55M  |
|[gray-Denoising](https://drive.google.com/drive/folders/1mKkctIvFYrpW88i1W-LzwtGqAMURoZem?usp=sharing)| 12 | 68.10M  |

### Data

This git repository also includes a set of test images and measurement matrices in the file `./data`(though this can be modified).  The datasets we evaluated on were Set11, BSD68, [50-Brain Images](https://github.com/jianzhangcs/ISTA-Net-PyTorch) and [CelebA HQ](https://github.com/tkarras/progressive_growing_of_gans).

### Run the Demo

to demonstrate the performance of PnP/RED for CS and CS-MRI, you can run the demo code by typing

```
$ python cs_mri.py
```

or

```
$ python cs_nature_color.py
```

or

```
$ python cs_nature_gray.py
```

The per iteration results will be stored in the ./results folder.



## Citation
J. Liu, M. S. Asif, B. Wohlberg, and U. S. Kamilov, “Recovery analysis for plug-and-play priors using the restricted eigenvalue condition,” Advances in Neural Information Processing Systems (NeurIPS), in press.
```
@inproceedings{liu2021recovery,
  author =	 {Jiaming Liu and M. Salman Asif and Brendt Wohlberg
		  and Ulugbek S. Kamilov},
  title =	 {Recovery Analysis for Plug-and-Play Priors using the
		  Restricted Eigenvalue Condition},
  year =	 2021,
  month =	 Dec,
  booktitle =	 {Advances in Neural Information Processing Systems
		  ({N}eur{IPS})},
  note =	 {in press}
}
```
