# BeLFusion
### Latent Diffusion for Behavior-Driven Human Motion Prediction (ICCV'23)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> 
[![arXiv](https://img.shields.io/badge/arXiv-2210.06551-b31b1b.svg)](https://arxiv.org/abs/2211.14304)
<a href="https://barquerogerman.github.io/BeLFusion/"><img alt="Project" src="https://img.shields.io/badge/-Project%20Page-lightgrey?logo=Google%20Chrome&color=informational&logoColor=white"></a>
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/belfusion-latent-diffusion-for-behavior/human-pose-forecasting-on-amass)](https://paperswithcode.com/sota/human-pose-forecasting-on-amass?p=belfusion-latent-diffusion-for-behavior)
<br>
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/belfusion-latent-diffusion-for-behavior/human-pose-forecasting-on-human36m)](https://paperswithcode.com/sota/human-pose-forecasting-on-human36m?p=belfusion-latent-diffusion-for-behavior)
<br>

![BeLFusion's architecture](assets/arch.png)

This repository contains the official PyTorch implementation of the paper:

**BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction**<br>
*German Barquero, Sergio Escalera, and Cristina Palmero*<br>
**ICCV 2023**<br>
[[website](https://barquerogerman.github.io/BeLFusion/)] [[paper](https://arxiv.org/abs/2211.14304)] [[demo](https://barquerogerman.github.io/BeLFusion/)]

**Note**: our data loaders consider an extra dimension for the number of people in the scene. Since the project aims at single-human motion prediction, this dimension is always 1.

## Installation


### 1. Environment

<details> 
<summary>OPTION 1 - Python/conda environment</summary>
<p>

```
conda create -n belfusion python=3.9.5
conda activate belfusion
pip install -r requirements.txt
```
</p>
</details> 

<details> 
<summary>OPTION 2 - Docker</summary>
<p>
We also provide a DockerFile to build a Docker image with all the required dependencies. 

**IMPORTANT**: This option will not let you launch the visualization script, as it requires a GUI. You will be able though to train and evaluate the models.

To build and launch the Docker image, run the following commands from the root of the repository:
```
docker build . -t belfusion
docker run -it --gpus all --rm --name belfusion \
-v ${PWD}:/project \
belfusion
```

You should now be in the container, ready to run the code.
</p>
</details> 


### 2. Datasets

#### [**> Human3.6M**](http://vision.imar.ro/human3.6m/description.php)
Extract the Poses-D3Positions* folders for S1, S5, S6, S7, S8, S9, S11 into `./datasets/Human36M`. Then, run:

```
python -m data_loader.parsers.h36m
```

#### [**> AMASS**](https://amass.is.tue.mpg.de/)
Download the *SMPL+H G* files for **22 datasets**: ACCAD, BMLhandball, BMLmovi, BMLrub, CMU, DanceDB, DFaust, EKUT, EyesJapanDataset, GRAB, HDM05, HUMAN4D, HumanEva, KIT, MoSh, PosePrior (MPI_Limits), SFU, SOMA, SSM, TCDHands, TotalCapture, and Transitions. Then, move the **tar.bz2** files to `./datasets/AMASS` (DO NOT extract them). 

Now, download the 'DMPLs for AMASS' from [here](https://smpl.is.tue.mpg.de), and the 'Extended SMPL+H model' from [here](https://mano.is.tue.mpg.de/). Move both extracted folders (dmpls, smplh) to `./auxiliar/body_models`. Then, run:

```
python -m data_loader.parsers.amass --gpu
```

**Note 1**: remove the `--gpu` flag if you do not have a GPU.

**Note 2**: this step could take a while (~2 hours in CPU, ~20-30 minutes in GPU).

### 3. Checkpoints [(link)](https://ubarcelona-my.sharepoint.com/:f:/g/personal/germanbarquero_ub_edu/EhInsrgQfe5OoqxBdHS21vcBxEJRU5JJq0zzmS2l8csc-A?e=LL1Guq)
Replace the folder 'checkpoints' in the root of the repository with the downloaded one. If you want to train the models from scratch, you can skip this step and go to the *training* section.


## Evaluation
Run the following scripts to evaluate BeLFusion and the other state-of-the-art methods.

Human3.6M:
```
# BeLFusion 
python eval_belfusion.py -c checkpoints/ours/h36m/BeLFusion/final_model/ -i 217 --ema --mode stats --batch_size 512

# Baselines --> {ThePoseKnows, DLow, GSPS, DiverseSampling}
python eval_baseline.py -c checkpoints/baselines/h36m/<BASELINE_NAME>/exp -m stats --batch_size 512
```

AMASS:
```
# BeLFusion
python eval_belfusion.py -c checkpoints/ours/amass/BeLFusion/final_model/ -i 1262 --multimodal_threshold 0.4 --ema --mode stats --batch_size 512

# Baselines --> {ThePoseKnows, DLow, GSPS, DiverseSampling}
python eval_baseline.py -c checkpoints/baselines/amass/<BASELINE_NAME>/exp -m stats --batch_size 512 --multimodal_threshold 0.4
```

- Add `--stats_mode all` to also compute the MMADE, MMFDE (increased computation time).
- Add `-cpu` to run the evaluation in CPU (recommended for low-memory GPUs).
- (only for BeLFusion) Use `--dstride S` to compute the evaluation metrics every S denoising steps (increased computation time). If S=10, the metrics will be computed for step 1 (BeLFusion_D), and 10 (BeLFusion).


## Visualization
Run the following scripts to visualize the results of BeLFusion and the other state-of-the-art methods (\<DATASET\> in {`h36m`, `amass`}).

```
# BeLFusion with Human3.6M (press '0' to visualize BeLFusion_D)
python eval_belfusion.py -c checkpoints/ours/h36m/BeLFusion/final_model/ -i 217 --ema --mode vis --batch_size 64 --dstride 10

# BeLFusion with AMASS (press '0' to visualize BeLFusion_D)
python eval_belfusion.py -c checkpoints/ours/amass/BeLFusion/final_model/ -i 1262 --ema --mode vis --batch_size 64 --dstride 10

# Baselines --> {ThePoseKnows, DLow, GSPS, DiverseSampling}
python eval_baseline.py -c checkpoints/baselines/<DATASET>/<BASELINE_NAME>/exp -m vis --batch_size 64
```

- Press `n` to navigate between the samples.
- Set `--samples N` to generate `N` samples. Set the columns in the visualization grid with `--ncols N`.
- During visualization, press `h` to show only the future motion (without observation).
- (only for BeLFusion) When `--dstride S` for S != -1, you can visualize the output of BeLFusion every `S` denoising steps (press keys `0`, `1`, `2`, ..., to navigate from 1, 1+S, 1+2S, ...).

**Note:** Replace `--mode vis` with `--mode gen` to generate the gif animations instead of visualizing them. In this mode, set the argument `--store_idx I` to store the gifs for denoising step `I`. For example, set `I` to 1 for BeLFusion_D's outputs.

## Training
For training BeLFusion from scratch, you need to first train the Behavioral Latent Space (BLS) and the observation autoencoder (\<DATASET\> in {`h36m`, `amass`}). Both models can be trained in parallel:
  
```
# Observation autoencoder --> 500 epochs
python train_auto.py -c checkpoints/ours/<DATASET>/BeLFusion/final_model/autoencoder_obs/config.json

# BLS --> 2x500 epochs
python train_bls.py -c checkpoints/ours/<DATASET>/BeLFusion/final_model/behavioral_latent_space/config.json
```

Once they finish, you can train the Latent Diffusion Model (LDM):

```
# BeLFusion --> 217/1262 epochs for H36M/AMASS
python train_belfusion.py -c checkpoints/ours/<DATASET>/BeLFusion/final_model/config.json
```

## Citation
If you find our work useful in your research, please consider citing our paper:
```
@article{barquero2023belfusion,
  title={BeLFusion: Latent Diffusion for Behavior-Driven Human Motion Prediction},
  author={Barquero, German and Escalera, Sergio and Palmero, Cristina},
  journal={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

## License
The software in this repository is freely available for free non-commercial use (see [license](https://github.com/BarqueroGerman/BeLFusion/blob/main/LICENSE) for further details).

**Note 1:** project structure borrowed from @victoresque's [template](https://github.com/victoresque/pytorch-template).

**Note 2:** code under `./models/sota` is based on the original implementations of the corresponding papers ([Dlow](https://github.com/Khrylx/DLow), [DiverseSampling](https://github.com/Droliven/diverse_sampling), and [GSPS](https://github.com/wei-mao-2019/gsps)).
