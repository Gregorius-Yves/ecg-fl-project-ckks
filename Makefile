COMPOSE = docker-compose
CKKS_ENV = CKKS_ENABLED=1

.PHONY: up up-ckks down build sidecar clients clients-ckks logs logs-ckks status clean

# Tanpa CKKS
up:
	$(COMPOSE) up -d --build dashboard

# Dengan CKKS encryption (server + sidecar)
up-ckks:
	$(COMPOSE) --profile ckks up -d --build ckks-sidecar
	$(CKKS_ENV) $(COMPOSE) up -d --build dashboard

# Klien tanpa CKKS
clients:
	$(COMPOSE) --profile clients up -d

# Klien dengan CKKS  (VM2=ptbxl, VM3=chapman, VM4=g12ec)
clients-ckks:
	$(CKKS_ENV) $(COMPOSE) --profile clients up -d

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
