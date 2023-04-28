#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker. The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.
image=$1
to_ecr=$2

if [ "${image}" == "" ]; then
  echo "Usage: $0 <image-name>"
  exit 1
fi

# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

# Get the region defined in the current configuration (default to us-west-2 if none defined)
region=$(aws configure get region)
region=${region:-us-west-2}
fullname="${account}.dkr.ecr.${region}.amazonaws.com/${image}:latest"

# If the repository doesn't exist in ECR, create it.
if ! aws ecr describe-repositories --repository-names "${image}" >/dev/null 2>&1; then
  aws ecr create-repository --repository-name "${image}" >/dev/null
fi

# Get the login command from ECR and execute it directly
aws --profile dev ecr get-login-password --region "${region}" | docker login --username AWS --password-stdin "${account}".dkr.ecr."${region}".amazonaws.com

# Build the docker image locally with the image name and then push it to ECR
# with the full name.
docker build --platform linux/amd64 -t "${image}" .
docker tag "${image}" "${fullname}"
if [ "$to_ecr" == "true" ]; then
  docker push "${fullname}"
fi
#
