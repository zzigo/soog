#!/usr/bin/env fish

echo "=== SOOG DEPLOY START ==="

set -l REPO_DIR /opt/soog
set -l BACKEND_DIR $REPO_DIR/backend
set -l FRONTEND_DIR $REPO_DIR/frontend
set -l OFFLOAD_DIR $BACKEND_DIR/offload
set -l BACKUP_ROOT /var/tmp/soog-deploy-backups
set -l TS (date "+%Y%m%d-%H%M%S")
set -l OFFLOAD_BACKUP "$BACKUP_ROOT/offload-$TS"

cd $REPO_DIR; or exit 1

# 1) Backup local generated artifacts (gallery/STL) before syncing code
if test -d $OFFLOAD_DIR
    echo "→ Backing up offload to $OFFLOAD_BACKUP"
    mkdir -p $OFFLOAD_BACKUP; or exit 1
    rsync -a "$OFFLOAD_DIR/" "$OFFLOAD_BACKUP/"; or exit 1
else
    echo "→ No offload directory found, skipping backup"
end

# 2) Force repository to remote code state (avoids pull conflicts with local lockfiles/artifacts)
echo "→ Syncing repository to origin/main"
git fetch origin main; or exit 1
git reset --hard origin/main; or exit 1
git clean -fd; or exit 1

# 3) Restore offload after code sync (merge local generated content back in)
if test -d $OFFLOAD_BACKUP
    echo "→ Restoring offload from backup"
    mkdir -p $OFFLOAD_DIR; or exit 1
    rsync -a "$OFFLOAD_BACKUP/" "$OFFLOAD_DIR/"; or exit 1
end

# 4) Backend dependencies (use venv pip directly)
echo "→ Updating backend"
set -gx PIP_DISABLE_PIP_VERSION_CHECK 1
set -l BACKEND_PIP "$BACKEND_DIR/venv/bin/pip"
if not test -x $BACKEND_PIP
    echo "✖ Missing backend virtualenv: $BACKEND_DIR/venv"
    exit 1
end
$BACKEND_PIP install -r "$BACKEND_DIR/requirements.txt"; or exit 1

# 5) Frontend clean install + build (lockfile-strict)
echo "→ Building frontend"
cd $FRONTEND_DIR; or exit 1
rm -rf node_modules .nuxt .output
if test -f package-lock.json
    npm ci; or exit 1
else
    npm install; or exit 1
end

set -l THREE_VERSION (node -p "require('./node_modules/three/package.json').version" 2>/dev/null)
if test -n "$THREE_VERSION"
    echo "→ three version: $THREE_VERSION"
end

node -e "const p=require('./node_modules/three/package.json'); if(!(p.exports&&p.exports['./webgpu'])){console.error('three missing ./webgpu export'); process.exit(2)}"; or exit 1
npm run build; or exit 1

# 6) Restart services
echo "→ Restarting services"
sudo systemctl restart soog-backend.service; or exit 1
sudo systemctl restart soog-frontend.service; or exit 1

echo "=== SOOG DEPLOY DONE ==="
