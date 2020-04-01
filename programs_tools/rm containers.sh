# command pour supprimer tous les conteneurs
docker container rm $(docker container ls -aq)