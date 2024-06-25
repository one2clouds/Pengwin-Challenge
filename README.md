python3 -m venv .venv
source .venv/bin/activate 
pip install -r ./requirements.txt

# Running Docker 
``` docker build -t docker_just_nnunet -f ./Dockerfile .```
``` docker run -it --rm --name docker_just_nnunet --gpus all --mount source=volume-just_nnunet,destination=/opt/app/output/ nvidia/cuda:11.0.3-base-ubuntu20.04 nvidia-smi ```
``` docker run -it --gpus all docker_just_nnunet ```


# Update the model inside the resources/ from this link:
https://drive.google.com/drive/folders/1-PA0aCnUO6SfIqH5gYml8mSGfm59k21a?usp=sharing
