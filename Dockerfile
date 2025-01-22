FROM pytorch/pytorch

WORKDIR /workspace

RUN pip install \
    docopt \
    pandas \
    transformers \
    scikit-learn \
    tqdm \
    wandb
