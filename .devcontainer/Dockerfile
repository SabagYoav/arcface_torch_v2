################################
# BASE IMAGE
################################

ARG BASE_IMAGE
FROM $BASE_IMAGE AS base

ARG ROS_DISTRO

# install ros packages
RUN sudo apt-get update &&\
    sudo apt-get install -y --no-install-recommends


###################################################
############### CREATE THE USER # #################
###################################################
ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

ENV USERNAME=$USERNAME

# Check if "ubuntu" user exists, delete it if it does, then create the desired user
RUN if getent passwd ubuntu > /dev/null 2>&1; then \
        userdel -r ubuntu && \
        echo "Deleted existing ubuntu user"; \
    fi && \
    groupadd --gid $USER_GID $USERNAME && \
    useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    echo "Created new user $USERNAME" && \
    usermod -aG root $USERNAME

# Add sudo support for the non-root user
RUN apt-get update && apt-get install -y sudo \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && rm -rf /var/lib/apt/lists/*


#### INSTALL FFMPEG #####
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


