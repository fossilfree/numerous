imageVersion=1.0.1
libraryImageName= mariuscristian/numerous-requirements:${imageVersion}
imageName = mariuscristian/numerous:${imageVersion}


install:
	pip3 install -e .

run-tests:
	python3 -m pytest

run-benchmark:
	python3 ./benchmark/tst.py 1000 100

benchmark:
	@echo python3 ./benchmark/tst.py $(filter-out $@,$(MAKECMDGOALS))

library-image:
	docker image build -t ${libraryImageName} - < Dockerfile_library
	docker push ${libraryImageName}

image-circle-ci:
	docker image build -t ${imageName}  .
	docker push ${imageName}

# Pushing an image to gcr instead of dockerhub:
# add tag to docker image
#docker tag <user-name>/<sample-image-name> gcr.io/<project-id>/<sample-image-name>:<tag>
# push image to gcloud container registry
#gcloud docker — push gcr.io/your-project-id/<project-id>/<sample-image-name>:<tag>


%:
	@:
