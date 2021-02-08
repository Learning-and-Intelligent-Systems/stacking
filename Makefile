# Makefile for Docker workflows with Franka Emika Panda Platform
# Copyright 2021 Massachusetts Institute of Technology

# Docker arguments
IMAGE_NAME = panda
DISPLAY ?= :0.0
XPASSTHROUGH ?= false
DOCKER_FILE_DIR = "."
DOCKERFILE = ${DOCKER_FILE_DIR}/Dockerfile
NUM_BUILD_CORES ?= 1

# Set Docker volumes and environment variables
HOST_RRG_DIR = ${HOME}/panda-docker
DOCKER_DEV_DIR = /catkin_ws/src/stacking
DOCKER_VOLUMES = \
	--volume="$(XAUTH):$(XAUTH)":rw \
	--volume="$(XSOCK)":$(XSOCK):rw \
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
	--volume="$(PWD)/stacking:/catkin_ws/src/stacking" \
	--volume="$(PWD)/devel:/catkin_ws/devel" \
	--volume="$(PWD)/build:/catkin_ws/build" \
	--volume="$(PWD)/logs:/catkin_ws/logs"
DOCKER_ENV_VARS = \
	--env="ROS_IP=127.0.0.1" \
	--env="NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics" \
	--env="DISPLAY=$(DISPLAY)" \
	--env="QT_X11_NO_MITSHM=1" \
	--env="XAUTHORITY=$(XAUTH)" \
	--env="XPASSTHROUGH=$(XPASSTHROUGH)"


###########
#  SETUP  #
###########
# Build the image incrementally using cache
.PHONY: build
build:
	@docker build -t ${IMAGE_NAME} \
		--build-arg NUM_BUILD_CORES=$(NUM_BUILD_CORES) \
		-f ./${DOCKERFILE} .

# Rebuild the image from scratch
.PHONY: rebuild
rebuild:
	@docker build -t ${IMAGE_NAME} --no-cache \
		--build-arg NUM_BUILD_CORES=$(NUM_BUILD_CORES) \
		-f ./${DOCKERFILE} .

# Set up local xhost sharing for visualization inside the container
# (e.g. Gazebo, RViz, matplotlib)
.PHONY: xhost-activate
xhost-activate:
	@echo "Enabling local xhost sharing:"
	@echo "  Display: $(DISPLAY)"
	@-DISPLAY=$(DISPLAY) xhost  +
	@- xhost  +
	@./setup_xauth.sh

# Kill any running Docker containers
.PHONY: kill
kill:
	@echo "Closing all running Docker containers:"
	@docker kill $(shell docker ps -q --filter ancestor=${IMAGE_NAME})


###########
#  TASKS  #
###########
# Start a terminal
.PHONY: term
term:
	@docker run -it --gpus all --net=host --ipc host --privileged \
		-w ${DOCKER_DEV_DIR} \
		${DOCKER_VOLUMES} ${DOCKER_ENV_VARS} \
		${IMAGE_NAME} bash
