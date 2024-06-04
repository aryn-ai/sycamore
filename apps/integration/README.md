# Container Integration Testing for Sycamore

To run these locally today:
```bash
cd sycamore
docker compose up reset
docker compose up -d
poetry run pytest apps/integration
```
