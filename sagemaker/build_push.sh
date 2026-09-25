#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${AWS_REGION:=us-east-1}"
: "${ECR_REPOSITORY:=ml-code-2}"
account_id="$(aws sts get-caller-identity --query Account --output text)"
registry="${account_id}.dkr.ecr.${AWS_REGION}.amazonaws.com"
if ! aws ecr describe-repositories --region "$AWS_REGION" --repository-names "$ECR_REPOSITORY" >/dev/null 2>&1; then
  aws ecr create-repository --region "$AWS_REGION" --repository-name "$ECR_REPOSITORY" >/dev/null
fi
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$registry"
image_uri="${registry}/${ECR_REPOSITORY}:$(date -u +%Y%m%d-%H%M%S)"
docker build --platform linux/amd64 -t "$image_uri" .
docker push "$image_uri"
echo "Training image: $image_uri"
