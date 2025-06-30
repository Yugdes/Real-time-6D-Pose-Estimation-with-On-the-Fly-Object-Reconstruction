# #!/bin/bash

# docker run --gpus all \
#     --ipc=host \
#     --env="DISPLAY=${DISPLAY}" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --device /dev/bus/usb \
#     # -v $(pwd):/workspace \
#     -v ~/yug_ws:/workspace \
#     -it foundationpose:latest
#!/bin/bash
set -e

docker run --gpus all \
    --ipc=host \
    --env="DISPLAY=${DISPLAY}" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --device /dev/bus/usb \
    -v ~/yug_ws:/workspace \
    -it foundationpose:latest \
    /bin/bash
