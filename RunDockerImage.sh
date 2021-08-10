IMAGE_REPOSITORY=my_mobile_face_net_tf_1_15__demo
IMAGE_TAG=latest
IMAGE_FULLNAME=${IMAGE_REPOSITORY}:${IMAGE_TAG}

# docker build if changed.
docker build -t ${IMAGE_FULLNAME} .

# create external dataset path
media_root="/media/$(whoami)"
media_name="$(ls ${media_root} | head -n 1)"
dataset_dir="${media_root}/${media_name}/dataset"

# allow display connection
# xhost local:
xhost +

# run container 
docker run \
--interactive \
--tty \
--rm \
--mount=type=bind,src="$(pwd)",dst=/root/share \
--mount=type=bind,src=/etc/group,dst=/etc/group,readonly \
--mount=type=bind,src=/etc/passwd,dst=/etc/passwd,readonly \
--mount=type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix \
$( if [ -e $HOME/.Xauthority ]; then echo "--mount=type=bind,src=$HOME/.Xauthority,dst=/root/.Xauthority"; fi ) \
$( if [ -e ${dataset_dir}    ]; then echo "--mount=type=bind,src=${dataset_dir},dst=/root/share/dataset" ; fi ) \
$( if [ -e /dev/video4       ]; then echo "--device=/dev/video4:/dev/video0"; else echo "--device=/dev/video0:/dev/video0"; fi ) \
--gpus=all \
--env=QT_X11_NO_MITSHM=1 \
--env=DISPLAY=${DISPLAY} \
--env=NVIDIA_DRIVER_CAPABILITIES=all \
--net=host \
--publish=8080:8080 \
--publish=8080:8080/udp \
--name=${IMAGE_REPOSITORY}_${IMAGE_TAG}_$(date "+%Y_%m%d_%H%M%S") \
${IMAGE_FULLNAME}