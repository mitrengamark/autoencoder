FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

LABEL maintainer="mitrengamark@edu.bme.hu"
LABEL docker_image_name="autoencoder"

SHELL ["/bin/bash", "-c"]

WORKDIR /root/autoencoder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install --no-install-recommends -qy \
        build-essential \
        autoconf \
        automake \
        sudo \
        vim \
        nano \
        git \
        curl \
        wget \
        tmux \
        openssh-server && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir /var/run/sshd && \
    echo 'root:root' | chpasswd && \
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication yes/' /etc/ssh/sshd_config

EXPOSE 22

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=graphics,utility,compute

RUN ln -s /opt/conda/bin/python /usr/bin/python3 || true

RUN echo "source /opt/conda/etc/profile.d/conda.sh && conda activate base" >> /root/.bashrc

COPY requirements.txt .

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

RUN useradd -m -s /bin/bash miti && \
    usermod -aG sudo miti

RUN mkdir -p /home/miti/autoencoder
COPY . /home/miti/autoencoder

RUN chown -R miti:miti /home/miti/autoencoder

# USER root
USER miti

WORKDIR /home/miti/autoencoder


# CMD ["/usr/sbin/sshd", "-D"]
CMD ["/bin/bash"]
