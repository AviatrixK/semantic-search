#!/usr/bin/env bash
# Update the production stack on the EC2 instance: git pull, build, up -d, show status.
# Run from anywhere:  ./scripts/deploy.sh        (add --no-pull to rebuild the checked-out code only)
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
DC=(docker compose --env-file .env.prod -f docker-compose.prod.yml)

[ -f .env.prod ] || { echo "ERROR: .env.prod not found. Copy .env.prod.example to .env.prod and fill it in." >&2; exit 1; }

# Fail early on the mistakes that are easiest to make.
if grep -q 'REPLACE_WITH' .env.prod; then
  echo "ERROR: .env.prod still contains REPLACE_WITH placeholders:" >&2
  grep -n 'REPLACE_WITH' .env.prod | sed 's/=.*/=.../' >&2
  exit 1
fi
pw=$(grep -E '^POSTGRES_PASSWORD=' .env.prod | head -1 | cut -d= -f2-)
if [ -z "$pw" ] || ! grep -E '^DATABASE_URL=' .env.prod | grep -qF "svs:${pw}@"; then
  echo "ERROR: DATABASE_URL in .env.prod must contain POSTGRES_PASSWORD (postgresql+psycopg://svs:<password>@postgres:5432/svs)." >&2
  exit 1
fi

if [ "${1:-}" != "--no-pull" ]; then
  git pull --ff-only
fi

"${DC[@]}" build
"${DC[@]}" up -d --remove-orphans

echo
echo "Waiting up to 3 minutes for the services to become healthy..."
for _ in $(seq 1 36); do
  if ! "${DC[@]}" ps --format '{{.Service}} {{.Health}}' | grep -Eq ' (starting|unhealthy)$'; then break; fi
  sleep 5
done
echo
"${DC[@]}" ps
echo
echo "Recent api log:"
"${DC[@]}" logs --tail 15 api
