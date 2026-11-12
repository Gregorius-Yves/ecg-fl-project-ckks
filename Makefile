COMPOSE = docker compose
CKKS_ENV = CKKS_ENABLED=1
SERVER_IP ?= 10.34.100.114

.PHONY: up up-ckks down build sidecar \
        clients clients-ckks \
        client-ptbxl client-chapman client-g12ec \
        client-ptbxl-ckks client-chapman-ckks client-g12ec-ckks \
        server server-ckks \
        logs logs-ckks logs-server logs-clients status clean

# ── Server (dashboard + Flower gRPC) ─────────────────────────────────

server:
	$(COMPOSE) up -d --build dashboard

server-ckks:
	$(COMPOSE) --profile ckks up -d --build ckks-sidecar
	$(CKKS_ENV) $(COMPOSE) up -d --build dashboard

# ── Semua klien sekaligus ─────────────────────────────────────────────

clients:
	$(COMPOSE) --profile clients up -d

clients-ckks:
	$(CKKS_ENV) $(COMPOSE) --profile clients up -d

# ── Klien individual (lokal, SERVER_ADDRESS=dashboard:8080) ───────────

client-ptbxl:
	$(COMPOSE) --profile clients up -d client-ptbxl

client-chapman:
	$(COMPOSE) --profile clients up -d client-chapman

client-g12ec:
	$(COMPOSE) --profile clients up -d client-g12ec

# ── Klien individual + CKKS ───────────────────────────────────────────

client-ptbxl-ckks:
	$(CKKS_ENV) $(COMPOSE) --profile clients up -d client-ptbxl

client-chapman-ckks:
	$(CKKS_ENV) $(COMPOSE) --profile clients up -d client-chapman

client-g12ec-ckks:
	$(CKKS_ENV) $(COMPOSE) --profile clients up -d client-g12ec

# ── Klien terdistribusi (dari VM lain, arahkan ke SERVER_IP) ──────────

client-ptbxl-remote:
	SERVER_ADDRESS=$(SERVER_IP):8080 $(COMPOSE) --profile clients up -d client-ptbxl

client-chapman-remote:
	SERVER_ADDRESS=$(SERVER_IP):8080 $(COMPOSE) --profile clients up -d client-chapman

client-g12ec-remote:
	SERVER_ADDRESS=$(SERVER_IP):8080 $(COMPOSE) --profile clients up -d client-g12ec

client-ptbxl-remote-ckks:
	SERVER_ADDRESS=$(SERVER_IP):8080 $(CKKS_ENV) $(COMPOSE) --profile clients up -d client-ptbxl

client-chapman-remote-ckks:
	SERVER_ADDRESS=$(SERVER_IP):8080 $(CKKS_ENV) $(COMPOSE) --profile clients up -d client-chapman

client-g12ec-remote-ckks:
	SERVER_ADDRESS=$(SERVER_IP):8080 $(CKKS_ENV) $(COMPOSE) --profile clients up -d client-g12ec

# ── Alias lama ────────────────────────────────────────────────────────

up: server
up-ckks: server-ckks

# ── Sidecar saja ──────────────────────────────────────────────────────

sidecar:
	$(COMPOSE) --profile ckks up -d --build ckks-sidecar

# ── Logs ──────────────────────────────────────────────────────────────

logs:
	$(COMPOSE) logs -f

logs-ckks:
	$(COMPOSE) logs -f ckks-sidecar

logs-server:
	$(COMPOSE) logs -f dashboard

logs-clients:
	$(COMPOSE) --profile clients logs -f

# ── Lifecycle ─────────────────────────────────────────────────────────

down:
	$(COMPOSE) --profile ckks --profile clients down

build:
	$(COMPOSE) build

status:
	$(COMPOSE) ps

clean:
	$(COMPOSE) --profile ckks --profile clients down -v --remove-orphans
