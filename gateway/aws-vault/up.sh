#!/usr/bin/env bash
# Start (or refresh) the gateway with temporary credentials from aws-vault.
#   ./up.sh                 whole stack
#   ./up.sh litellm         one service; any `docker compose up` arguments work
# Credentials are fixed when the containers start and last AWS_VAULT_DURATION (12h, the
# role's maximum). When they expire Bedrock calls fail; run this again to replace them.
set -euo pipefail
cd "$(dirname "$0")/.."
exec aws-vault exec "${AWS_VAULT_PROFILE:-bedrock-invoke}" --duration="${AWS_VAULT_DURATION:-12h}" -- \
  docker compose -f docker-compose.yml -f aws-vault/docker-compose.aws-vault.yml up -d "$@"
