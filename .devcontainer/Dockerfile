################################
# BASE IMAGE
################################

ARG BASE_IMAGE
FROM $BASE_IMAGE AS base

ARG ROS_DISTRO

# install ros packages
RUN sudo apt-get update &&\
    sudo apt-get install -y --no-install-recommends\
    ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-dev-tools

# install pip packages
RUN pip3 install \
    opencv-python-headless \
    numpy \ 
    shapely \ 
    matplotlib \ 
    jsonpickle \ 
    dicttoxml \ 
    scipy \ 
    ultralytics \
    lap \ 
    cython_bbox \
    yacs \
    termcolor


RUN pip3 install --no-dependencies \
	--extra-index-url http://gitea.ironbrain.internal:3000/api/packages/IronBrain/pypi/simple/ --trusted-host gitea.ironbrain.internal \
  ib_detector \
  ib_general \
  ib_indexer 


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

# Set up autocompletion for user
RUN apt-get update && apt-get install -y git-core bash-completion \
  && echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc \
  && echo "if [ -f /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash ]; then source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash; fi" >> /home/$USERNAME/.bashrc \
  && rm -rf /var/lib/apt/lists/*
    
# ##### FASTRTPS DDS #####
RUN echo export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp >> /home/$USERNAME/.bashrc && \
    export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp 

# #### SOURSE ROS2 ###
RUN echo "if [ -f /opt/ros/${ROS_DISTRO}/setup.bash ]; then source /opt/ros/${ROS_DISTRO}/setup.bash; fi" >> /home/$USERNAME/.bashrc

# ##### SET LD LIBRARY PATH #####
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:$LD_LIBRARY_PATH

# ensure the runtime shell also has it for the non-root user’s terminals
RUN echo 'export PYTHONPATH="/workspaces/ib_tracker/:${PYTHONPATH}"' >> /home/$USERNAME/.bashrc && \
    chown $USERNAME:$USERNAME /home/$USERNAME/.bashrc

#### INSTALL FFMPEG #####
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# ###################################################
# ############### DEVELOPMENT IMAGE #################
# #### install extra tools only for development #####
# ###################################################
FROM base as dev

RUN apt-get update &&\
    apt-get install -y \
    gdb \
    openssh-client \
    # ros-dev-tools \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


