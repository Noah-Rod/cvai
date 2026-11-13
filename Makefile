.PHONY: help start stop restart status logs open-frontend

COMPOSE ?= docker compose
FRONTEND_URL ?= http://localhost:8501

ifeq ($(OS),Windows_NT)
BROWSER_OPEN = powershell -NoProfile -Command "Start-Process '$(FRONTEND_URL)'"
else
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
BROWSER_OPEN = open "$(FRONTEND_URL)"
else
BROWSER_OPEN = xdg-open "$(FRONTEND_URL)"
endif
endif

help:
	@echo "Available commands:"
	@echo "  make start    Build and start the app in the background"
	@echo "  make stop     Stop and remove the app containers"
	@echo "  make restart  Restart the app containers"
	@echo "  make status   Show container status"
	@echo "  make logs     Follow app logs"

start:
	$(COMPOSE) up -d --build
	$(MAKE) open-frontend

stop:
	$(COMPOSE) down

restart:
	$(COMPOSE) restart
	$(MAKE) open-frontend

open-frontend:
	$(BROWSER_OPEN)

status:
	$(COMPOSE) ps

logs:
	$(COMPOSE) logs -f
