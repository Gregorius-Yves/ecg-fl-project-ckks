COMPOSE = docker-compose
CKKS_ENV = CKKS_ENABLED=1
CLIENT ?= client-ptbxl

.PHONY: up up-ckks down build sidecar logs logs-ckks status clean

# Tanpa CKKS
up:
	$(COMPOSE) up -d --build dashboard
	$(COMPOSE) --profile clients up -d $(CLIENT)

# Dengan CKKS encryption
up-ckks:
	$(COMPOSE) --profile ckks up -d --build ckks-sidecar
	$(CKKS_ENV) $(COMPOSE) up -d --build dashboard
	$(CKKS_ENV) $(COMPOSE) --profile clients up -d $(CLIENT)

down:
	$(COMPOSE) --profile ckks --profile clients down

build:
	$(COMPOSE) build

sidecar:
	$(COMPOSE) --profile ckks up -d --build ckks-sidecar

logs:
	$(COMPOSE) logs -f

logs-ckks:
	$(COMPOSE) logs -f ckks-sidecar

status:
	$(COMPOSE) ps

clean:
	$(COMPOSE) --profile ckks --profile clients down -v --remove-orphans
