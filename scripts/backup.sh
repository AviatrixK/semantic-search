#!/usr/bin/env bash
# Dump the Postgres database to backups/svs-YYYYmmdd-HHMMSS.dump (custom format), keep the newest 7.
#   ./scripts/backup.sh            local file only
#   ./scripts/backup.sh --s3       also copy it to s3://$S3_BUCKET/backups/ (needs the AWS CLI on the host and
#                                  s3:PutObject on backups/* in the instance role)
# Restore: see docs/DEPLOY.md. Videos are not in the dump: they already live in S3.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DC=(docker compose --env-file .env.prod -f docker-compose.prod.yml)
KEEP=7
DIR=backups

mkdir -p "$DIR"
file="$DIR/svs-$(date -u +%Y%m%d-%H%M%S).dump"

# Write to a temp name first so a failed dump never looks like a good backup.
"${DC[@]}" exec -T postgres pg_dump -U svs -d svs --format=custom --no-owner > "$file.partial"
mv "$file.partial" "$file"
echo "Wrote $file ($(du -h "$file" | cut -f1))"

# Keep only the newest $KEEP dumps.
ls -1t "$DIR"/svs-*.dump 2>/dev/null | tail -n +$((KEEP + 1)) | xargs -r rm -f --

if [ "${1:-}" = "--s3" ]; then
  command -v aws >/dev/null || { echo "ERROR: aws CLI not installed (sudo snap install aws-cli --classic)" >&2; exit 1; }
  bucket=$(grep -E '^S3_BUCKET=' .env.prod | head -1 | cut -d= -f2-)
  region=$(grep -E '^AWS_REGION=' .env.prod | head -1 | cut -d= -f2-)
  aws s3 cp "$file" "s3://${bucket}/backups/$(basename "$file")" --region "${region:-ap-south-1}"
  echo "Copied to s3://${bucket}/backups/$(basename "$file") (old copies in S3 are not pruned: add a lifecycle rule)"
fi
