#!/usr/bin/env fish

echo "=== SOOG DEPLOY START ==="

# 1. Git pull
echo "→ Updating repository"
cd /opt/soog || exit 1
git pull || exit 1

# 2. Backend
echo "→ Updating backend"
source /opt/soog/backend/venv/bin/activate.fish
pip install -r /opt/soog/backend/requirements.txt || exit 1
deactivate

# 3. Frontend
echo "→ Building frontend"
cd /opt/soog/frontend || exit 1
bun run build || exit 1

# 4. Restart services
echo "→ Restarting services"
sudo systemctl restart soog-backend.service || exit 1
sudo systemctl restart soog-frontend.service || exit 1

echo "=== SOOG DEPLOY DONE ==="