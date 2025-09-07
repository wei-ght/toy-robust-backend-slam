FROM navifra/ros:humble-lab

ENV TZ=Asia/Seoul

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y \
  libomp-dev \
  clang \
  clang-format \
  gdb \
  libcoin-dev \
  python3-tk \
  locales && \
  locale-gen en_US.UTF-8 && \
  update-locale LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 && \
  pip3 install redis GitPython debugpy mplcursors

# RUN sudo mkdir -p /etc/apt/keyrings
# RUN curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
# RUN echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo `lsb_release -cs` main" | \
#   sudo tee /etc/apt/sources.list.d/librealsense.list
# RUN sudo apt-get update

# WORKDIR /root
# RUN git clone https://github.com/Microsoft/vcpkg.git

# WORKDIR /root/vcpkg
# RUN sudo ./bootstrap-vcpkg.sh \
#   && ./vcpkg integrate install \
#   && ./vcpkg install realsense2


RUN echo "source /opt/ros/humble/setup.bash"  >>  /root/.bashrc
RUN curl https://packages.osrfoundation.org/gazebo.gpg --output /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/gazebo-stable.list > /dev/null

RUN sudo apt-get update
RUN sudo apt-get install -y ignition-fortress
RUN sudo apt-get install -y ros-humble-ros-gzharmonic
RUN sudo apt-get install -y ros-humble-tf-transformations

# use for ros2 dds
RUN sudo apt-get install -y ros-humble-rmw-cyclonedds-cpp  

ARG USERNAME=navifra
ARG PASSWORD=1234
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
  && apt-get update \
  && apt-get install -y sudo wget \
  && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
  && chmod 0440 /etc/sudoers.d/$USERNAME \
  && apt-get autoremove -y \
  && apt-get clean -y \
  && rm -rf /var/lib/apt/lists/* \
  && chsh -s /usr/bin/zsh ${USERNAME}

# COPY postcreated.sh /usr/local/bin/postcreated.sh
# RUN chmod +x /usr/local/bin/postcreated.sh

USER $USERNAME
WORKDIR /home/${USERNAME}/colcon_ws

COPY zsh-in-docker.sh /tmp
RUN /tmp/zsh-in-docker.sh \
  -t https://github.com/denysdovhan/spaceship-prompt \
  -a 'SPACESHIP_PROMPT_ADD_NEWLINE="false"' \
  -a 'SPACESHIP_PROMPT_SEPARATE_LINE="false"' \
  -p git \
  -p https://github.com/zsh-users/zsh-autosuggestions \
  -p https://github.com/zsh-users/zsh-completions \
  -p https://github.com/zsh-users/zsh-history-substring-search \
  -p https://github.com/zsh-users/zsh-syntax-highlighting \
  -p 'history-substring-search'

RUN echo "export REDIS_HOST="host.docker.internal"" >> /home/$USERNAME/.bashrc
RUN echo "export BACKEND_HOST="host.docker.internal"" >> /home/$USERNAME/.bashrc
RUN echo "export REDIS_HOST="host.docker.internal"" >> /home/$USERNAME/.zshrc
RUN echo "export BACKEND_HOST="host.docker.internal"" >> /home/$USERNAME/.zshrc

RUN echo "export ROBOT_IP="localhost"" >> /home/$USERNAME/.bashrc
RUN echo "export ROBOT_IP="localhost"" >> /home/$USERNAME/.zshrc
RUN echo "export GTEST_COLOR=1" >> /home/$USERNAME/.bashrc
RUN echo "export GTEST_COLOR=1" >> /home/$USERNAME/.zshrc

RUN echo "source /opt/ros/humble/setup.bash" >> /home/$USERNAME/.bashrc
RUN echo "export USE_NO_PASS_REDIS=true" >> /home/$USERNAME/.bashrc
RUN echo "source /opt/ros/humble/setup.zsh" >> /home/$USERNAME/.zshrc
RUN echo "source ~/colcon_ws/install/setup.zsh" >> /home/$USERNAME/.zshrc
RUN echo "export USE_NO_PASS_REDIS=true" >> /home/$USERNAME/.zshrc

RUN echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/$USERNAME/.zshrc 

# RUN echo "ZSH_THEME=\"agnoster\"" >> /home/$USERNAME/.zshrc
RUN sed -i 's/spaceship-prompt\/spaceship/agnoster/g' /home/$USERNAME/.zshrc
RUN sed -i 's/zsh-completions//g' /home/$USERNAME/.zshrc
RUN sed -i 's/zsh-history-substring-search//g' /home/$USERNAME/.zshrc
RUN sed -i 's/history-substring-search//g' /home/$USERNAME/.zshrc

# RUN sudo apt-get install -y libglm-dev libglfw3-dev
# RUN sudo apt update && sudo apt install -y libspdlog-dev libpoco-dev

# RUN sudo apt-get install -y python3-tk
# RUN pip install mplcursors

RUN echo 'debconf debconf/frontend select Noninteractive' | sudo debconf-set-selections && \
  echo 'keyboard-configuration keyboard-configuration/layout select English (US)' | sudo debconf-set-selections && \
  echo 'keyboard-configuration keyboard-configuration/model select Generic 105-key PC (intl.)' | sudo debconf-set-selections && \
  echo 'keyboard-configuration keyboard-configuration/variant select English (US)' | sudo debconf-set-selections && \
  echo 'keyboard-configuration keyboard-configuration/options select ' | sudo debconf-set-selections && \
  DEBIAN_FRONTEND=noninteractive sudo apt update && \
  DEBIAN_FRONTEND=noninteractive sudo apt install -y \
  libgtk-3-dev \
  libqt5widgets5 \
  zenity \
  kdialog \
  libfmt-dev \
  keyboard-configuration devscripts debhelper dh-make lintian fakeroot equivs python3-bloom

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash

RUN source ~/.zshrc && nvm install node && npm install -g @anthropic-ai/claude-code

RUN sudo ldconfig

