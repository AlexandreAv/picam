# command pour supprimer tous les conteneurs
docker container rm -f $(docker container ls -aq)