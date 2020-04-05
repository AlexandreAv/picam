# command pour lancer le container
docker run -P --runtime=nvidia --gpus=0 --rm -v "/mnt/storage/Users/rrang/Documents/Programmation/Projets/Python/Projet Picam/:/app" -v "/tmp/.X11-unix:/tmp/.X11-unix" -v "/home/rangom/.Xauthority:/root/.Xauthority" --env DISPLAY=unix:0 --env NVIDIA_VISIBLE_DEVICES=all --env NVIDIA_DRIVER_CAPABILITIES=compute,utility --device=/dev/video0:/dev/video0 --net host alexandreav/tf_opencv python3 picam/main.py