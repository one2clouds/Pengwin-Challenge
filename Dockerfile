FROM --platform=linux/amd64 pytorch/pytorch 

ENV PYTHONBUFFER 1
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

USER user 

WORKDIR /opt/app

# Note : All the files that you interact with should be within the directory you specify for example if you write code :
# docker build -t docker_just_nnunet -f ./PENGWIN-example-algorithm/PENGWIN-challenge-packages/preliminary-development-phase-ct/Dockerfile . 
# Notice the . (dot) at last, it gives main directory as PENGWIN-example-algorithm and it will be considered your root folder and you need to keep files within that folder 

# RUN git clone https://github.com/MIC-DKFZ/nnUNet.git

COPY --chown=user:user ./requirements.txt /opt/app/
# /home/shirshak/Just-nnUNet-not-Overridden-with-FracSegNet-in-venv/PENGWIN-example-algorithm/PENGWIN-challenge-packages/preliminary-development-phase-ct/resources/last.ckpt

COPY --chown=user:user ./resources/best.ckpt /opt/app/resources/
# COPY --chown=user:user ./resources/model_best.model /opt/app/resources/
COPY --chown=user:user ./resources/model_best.model.pkl /opt/app/resources/
# RUN cp ./last.ckpt /opt/app/resources
# RUN cp ./model_best.model.pkl /opt/app/resources

RUN mkdir -p /opt/app/input/images/pelvic-fracture-ct/
# COPY Sample ct image for test
# COPY --chown=user:user ./test/input/*.mha /opt/app/input/images/pelvic-fracture-ct/

# COPY --chown=user:user ./dataset/

RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

RUN python -m pip install gdown 
RUN python -m gdown https://drive.google.com/uc?id=1TDlfk8tGhMRIvk86nG8yspna2ZZ1-Lf0 -O /opt/app/resources/

# RUN python -m pip install 'monai[all]==1.1.0'

COPY --chown=user:user ./split_img_to_SA_LI_RI.py /opt/app/
COPY --chown=user:user ./inference_docker.py /opt/app/

# COPY NEEDED FILES FOR LOADING 1ST MODEL 
COPY --chown=user:user ./base_module.py /opt/app/src/models/
COPY --chown=user:user ./unet.py /opt/app/src/models/

ENTRYPOINT ["python", "inference_docker.py"]




# docker build -t docker_just_nnunet -f ./Dockerfile .
# docker run -it --rm --name docker_just_nnunet --gpus all --mount source=volume-just_nnunet,destination=/opt/app/output/ nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi
# docker run -it --gpus all docker_just_nnunet








