#!/bin/sh -ex

DOCKERFILE_PATH=${1}
IMAGE_NAME=${2}
FULL_IMAGE_NAME="${CI_REGISTRY_IMAGE}/${IMAGE_NAME}"

echo "CI_COMMIT_SHORT_SHA=${CI_COMMIT_SHORT_SHA}"
echo "CI_PROJECT_DIR=${CI_PROJECT_DIR}"
echo "CI_PROJECT_ID=${CI_PROJECT_ID}"
echo "FULL_IMAGE_NAME=${FULL_IMAGE_NAME}"

date_commit_tag="$(date -I)-${CI_COMMIT_SHORT_SHA}"
echo "date_commit_tag=${date_commit_tag}"

docker pull "${FULL_IMAGE_NAME}":latest || true

docker build "${CI_PROJECT_DIR}" \
    -f "${DOCKERFILE_PATH}" \
    --cache-from "${FULL_IMAGE_NAME}":latest \
    --tag "${FULL_IMAGE_NAME}":"${date_commit_tag}" \
    --tag "${FULL_IMAGE_NAME}:latest"

docker push "${FULL_IMAGE_NAME}:${date_commit_tag}"
docker push "${FULL_IMAGE_NAME}:latest"

if [ -n "${CI_COMMIT_TAG}" ]; then
    docker tag "${FULL_IMAGE_NAME}":"${date_commit_tag}" "${FULL_IMAGE_NAME}:${CI_COMMIT_TAG}"
    docker push "${FULL_IMAGE_NAME}:${CI_COMMIT_TAG}"
fi
