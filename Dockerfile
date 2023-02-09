ARG PYTORCH="1.9.1"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN export TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.0+PTX" \
	&& export TORCH_NVCC_FLAGS="-Xfatbin -compress-all" \
	&& export CMAKE_PREFIX_PATH="$(dirname $(which conda))/../" 

# INIT OF FIX FOR GPG KEY -- Hopefully they will fix this soon
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# END OF FIX FOR GPG KEY

# apt and apt-get update
RUN apt-get update && apt-get install -y apt-transport-https && apt update && apt upgrade -y && apt-get clean
# apt and apt-get installations
RUN apt install -y vim git unzip \
	&& apt-get install -y libosmesa6-dev build-essential libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libopenmpi-dev tmux wget libglib2.0-0

ENV CONDA_DIR /opt/conda
ENV PATH $CONDA_DIR/bin:$PATH

RUN echo "Installing basic conda environment for PyTorch..." \
	&& conda create --name main python=3.9.5 \
	&& . /opt/conda/etc/profile.d/conda.sh \
	&& conda activate main \
	&& conda install numpy \
	&& pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html \
	&& pip install opencv-contrib-python pyyaml yacs tqdm colorama matplotlib cython tensorboardX imageio imageio-ffmpeg munkres pandas seaborn h5py sklearn 

RUN echo "Installing other dependencies for BeLFusion..." \
	&& . /opt/conda/etc/profile.d/conda.sh \
	&& conda activate main \
	&& pip install matplotlib zarr jupyterlab ipywidgets ml-collections==0.1.0 tensorboard>=2.4.0 absl-py==0.10.0 ninja mpi4py einops timm setuptools==59.5.0 cdflib scipy GitPython

ENV PROJECT_PATH=/project
WORKDIR $PROJECT_PATH

RUN echo ". /opt/conda/etc/profile.d/conda.sh  && conda activate main" > ~/.bashrc
