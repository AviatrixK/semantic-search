# Production shortcuts. Run on the EC2 server from the repo root:  make ps | logs | deploy | shell-api
# Every docker compose call uses the production env file and compose file.
DC := docker compose --env-file .env.prod -f docker-compose.prod.yml

.PHONY: ps logs deploy shell-api

ps:  ## container status and health
	$(DC) ps

# make logs                 follow every service
# make logs s=worker        follow one service (api, worker, beat, caddy, postgres, redis)
logs:  ## follow logs (last 100 lines first)
	$(DC) logs -f --tail=100 $(s)

# scripts/deploy.sh does: checks .env.prod, git pull --ff-only, `$(DC) build`, `$(DC) up -d --remove-orphans`, waits for health.
deploy:  ## pull, build, restart, wait until healthy
	./scripts/deploy.sh

shell-api:  ## shell inside the api container
	$(DC) exec api bash
