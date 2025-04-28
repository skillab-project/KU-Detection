FROM python:3.11-bullseye
COPY . /app/.
WORKDIR /app

# Install dependencies, including Git and Git LFS
RUN apt-get update && apt-get install -y \
    git \
    curl && \
    apt-get clean

# Install Git LFS from the official source
RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    git lfs install --skip-repo

# Get model
RUN mkdir -p models/codebert && \
    git clone https://huggingface.co/nnikolaidis/java-ku models/codebert && \
    cd models/codebert && git lfs pull && \
    rm -rf .git

RUN pip install -r /app/requirements.txt