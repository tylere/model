FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu118-ubuntu20.04-ec2
# https://github.com/aws/deep-learning-containers/blob/master/available_images.md

RUN pip install \
  einops \
  geopandas \
  jupyterlab \
  jsonargparse[signatures]>=4.27.7 \
  lightning \
  matplotlib \
  python-box \
  pyarrow \
  rasterio \
  s3fs \
  scikit-image \
  scikit-learn \
  torch~=2.1.0 \
  torchdata \
  torchvision \
  timm \
  vit-pytorch~=1.6.4 \
  wandb \
  https://github.com/Clay-foundation/stacchip/archive/refs/tags/0.1.29.zip \
  s3fs

ADD src /clay/src
ADD configs /clay/configs

ENV PYTHONPATH=/clay

WORKDIR /clay
RUN pip install https://github.com/Clay-foundation/stacchip/archive/refs/tags/0.1.30.zip
ADD nbs/naip-embeddings.py /clay/src/naip-embeddings.py
ADD nbs/wc-embeddings.py /clay/src/wc-embeddings.py
