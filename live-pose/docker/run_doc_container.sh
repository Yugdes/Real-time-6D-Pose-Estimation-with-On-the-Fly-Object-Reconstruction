docker rm -f foundationpose
DIR=$(pwd)/
docker run --gpus all --privileged -it \
  --name foundationpose \
  -v ~/yug_ws:/workspace \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  foundationpose:cuda-fixed

