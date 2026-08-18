#!/usr/bin/env fish

echo "=== SOOG DEPLOY START ==="

set -l REPO_DIR /opt/soog
set -l BACKEND_DIR $REPO_DIR/backend
set -l FRONTEND_DIR $REPO_DIR/frontend
set -l OFFLOAD_DIR_BACKEND $BACKEND_DIR/offload
set -l OFFLOAD_DIR_ROOT $REPO_DIR/offload
set -l OFFLINE_DIR_ROOT $REPO_DIR/offline
set -l SOOGI_DIR_ROOT $REPO_DIR/soogi
set -l BACKUP_ROOT /var/tmp/soog-deploy-backups
set -l TS (date "+%Y%m%d-%H%M%S")
set -l BACKUP_DIR "$BACKUP_ROOT/deploy-backup-$TS"

cd $REPO_DIR; or exit 1

# 1) Backup local generated artifacts (gallery/STL/audio) before syncing code
echo "→ Backing up persistent data to $BACKUP_DIR"
mkdir -p "$BACKUP_DIR/backend"; or exit 1

if test -d $OFFLOAD_DIR_BACKEND
    echo "  - Backing up backend/offload"
    rsync -rtv --no-o --no-g "$OFFLOAD_DIR_BACKEND/" "$BACKUP_DIR/backend/offload/"; or exit 1
end

if test -d $OFFLOAD_DIR_ROOT
    echo "  - Backing up root offload"
    rsync -rtv --no-o --no-g "$OFFLOAD_DIR_ROOT/" "$BACKUP_DIR/offload/"; or exit 1
end

if test -d $OFFLINE_DIR_ROOT
    echo "  - Backing up root offline"
    rsync -rtv --no-o --no-g "$OFFLINE_DIR_ROOT/" "$BACKUP_DIR/offline/"; or exit 1
end

if test -d $SOOGI_DIR_ROOT
    echo "  - Backing up root soogi"
    rsync -rtv --no-o --no-g "$SOOGI_DIR_ROOT/" "$BACKUP_DIR/soogi/"; or exit 1
end

# 2) Force repository to remote code state
echo "→ Syncing repository to origin/main"
git fetch origin main; or exit 1
git reset --hard origin/main; or exit 1
git clean -fd; or exit 1

# 3) Restore persistent data after code sync
echo "→ Restoring persistent data from backup"
if test -d "$BACKUP_DIR/backend/offload"
    mkdir -p $OFFLOAD_DIR_BACKEND; or exit 1
    rsync -rtv --no-o --no-g "$BACKUP_DIR/backend/offload/" "$OFFLOAD_DIR_BACKEND/"; or exit 1
end

if test -d "$BACKUP_DIR/offload"
    mkdir -p $OFFLOAD_DIR_ROOT; or exit 1
    rsync -rtv --no-o --no-g "$BACKUP_DIR/offload/" "$OFFLOAD_DIR_ROOT/"; or exit 1
end

if test -d "$BACKUP_DIR/offline"
    mkdir -p $OFFLINE_DIR_ROOT; or exit 1
    rsync -rtv --no-o --no-g "$BACKUP_DIR/offline/" "$OFFLINE_DIR_ROOT/"; or exit 1
end

if test -d "$BACKUP_DIR/soogi"
    mkdir -p $SOOGI_DIR_ROOT; or exit 1
    rsync -rtv --no-o --no-g "$BACKUP_DIR/soogi/" "$SOOGI_DIR_ROOT/"; or exit 1
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
sudo rm -rf node_modules .nuxt .output
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
echo "→ Restarting runtime"
set -l USED_PM2 0
set -l APP_USER (stat -c %U $REPO_DIR 2>/dev/null)
if test -z "$APP_USER"
    set APP_USER (whoami)
end
set -l APP_HOME (getent passwd $APP_USER | cut -d: -f6)
if test -z "$APP_HOME"
    set APP_HOME "/home/$APP_USER"
end
set -l PM2_HOME_DIR "$APP_HOME/.pm2"

if command -sq pm2
    set -l PM2_BIN (command -s pm2)
    set -l PM2_PREFIX
    if test "$APP_USER" != (whoami)
        set PM2_PREFIX sudo -u $APP_USER -H env PM2_HOME="$PM2_HOME_DIR" PATH="/usr/local/bin:/usr/bin:/bin"
    end

    if $PM2_PREFIX $PM2_BIN describe soog-backend >/dev/null 2>/dev/null
        echo "  - Restarting PM2 app ($APP_USER): soog-backend"
        $PM2_PREFIX $PM2_BIN restart soog-backend --update-env; or exit 1
        set USED_PM2 1
    end

    if $PM2_PREFIX $PM2_BIN describe soog-frontend >/dev/null 2>/dev/null
        echo "  - Restarting PM2 app ($APP_USER): soog-frontend"
        $PM2_PREFIX $PM2_BIN restart soog-frontend --update-env; or exit 1
        set USED_PM2 1
    end

    if test $USED_PM2 -eq 1
        echo "  - Saving PM2 process list for $APP_USER"
        $PM2_PREFIX $PM2_BIN save; or exit 1
    end
end

if test $USED_PM2 -eq 0
    echo "  - PM2 apps not found, falling back to systemd"
    if systemctl list-unit-files --type=service | grep -q '^soog-backend\.service'
        sudo systemctl restart soog-backend.service; or exit 1
    else
        echo "✖ soog-backend not found in PM2 or systemd"
        exit 1
    end

    if systemctl list-unit-files --type=service | grep -q '^soog-frontend\.service'
        sudo systemctl restart soog-frontend.service; or exit 1
    else
        echo "✖ soog-frontend not found in PM2 or systemd"
        exit 1
    end
end

# 7) Health checks
echo "→ Verifying local endpoints"
sleep 3
curl -fsS http://127.0.0.1:10000/api/ollama/verify >/dev/null; or exit 1
curl -fsSI http://127.0.0.1:3000 >/dev/null; or exit 1

echo "=== SOOG DEPLOY DONE ==="
