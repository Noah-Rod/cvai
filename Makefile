.PHONY: up down logs dev test lint fmt live live-api train-smoke

up:            ## build & start api + frontend
	docker compose up -d --build
down:
	docker compose down
logs:
	docker compose logs -f
dev:           ## start with file sync / hot reload
	docker compose watch
test:
	uv run --extra dev pytest -q
lint:
	uv run --extra dev ruff check src tools tests
fmt:
	uv run --extra dev ruff format src tools tests
live:          ## live webcam detection with local weights
	uv run --extra live python tools/live_detect.py --source 0
live-api:      ## live webcam detection through the running API
	uv run --extra live python tools/live_detect.py --source 0 --api http://localhost:8000
train-smoke:   ## 1-epoch CPU training run in docker
	docker compose --profile train run --rm train
