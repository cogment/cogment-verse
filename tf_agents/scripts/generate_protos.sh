#!/usr/bin/env bash

set -o errexit

PACKAGE_DIR="$(dirname "${BASH_SOURCE[0]}")/.."
COGMENT_API_VERSION="v1.2.1"
COGMENT_SITE_PACKAGES_DIR=$(dirname "$(dirname "$(python -c"import cogment; print(cogment.__file__)")")")

cd "${PACKAGE_DIR}"

# Overriding the cogment api version that comes with the python SDK
curl -s -L "https://cogment.github.io/cogment-api/${COGMENT_API_VERSION}/cogment-api-${COGMENT_API_VERSION}.tar.gz" | tar xz -C "${COGMENT_SITE_PACKAGES_DIR}/cogment/api"

python -m grpc.tools.protoc --proto_path="${COGMENT_SITE_PACKAGES_DIR}" \
  --python_out="${COGMENT_SITE_PACKAGES_DIR}" --grpc_python_out="${COGMENT_SITE_PACKAGES_DIR}" \
  "${COGMENT_SITE_PACKAGES_DIR}/cogment/api/model_registry.proto" \
  "${COGMENT_SITE_PACKAGES_DIR}/cogment/api/trial_datastore.proto"

# Generate cogment_verse protos
python -m cogment.generate
