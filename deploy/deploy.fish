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

function run_pm2_candidate
    set -l candidate $argv[1]
    set -l candidate_home $argv[2]
    set -e argv[1..2]

    if test "$candidate" = (whoami)
        $PM2_BIN $argv
    else
        sudo -u $candidate -H env PM2_HOME="$candidate_home/.pm2" PATH="/usr/local/bin:/usr/bin:/bin" $PM2_BIN $argv
    end
end

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

if command -sq pm2
    set -l PM2_BIN (command -s pm2)
    set -l PM2_USERS
    set -l REPO_OWNER (stat -c %U $REPO_DIR 2>/dev/null)
    set -l PM2_DAEMON_OWNER (ps -eo user=,cmd= | awk '/PM2 v[0-9.]+: God Daemon/ {print $1; exit}')

    for candidate in (whoami) $SUDO_USER $REPO_OWNER $PM2_DAEMON_OWNER zz
        if test -n "$candidate"
            if not contains -- $candidate $PM2_USERS
                set -a PM2_USERS $candidate
            end
        end
    end

    set -l PM2_APP_USER
    set -l PM2_PREFIX
    for candidate in $PM2_USERS
        set -l CANDIDATE_HOME (getent passwd $candidate | cut -d: -f6)
        if test -z "$CANDIDATE_HOME"
            set CANDIDATE_HOME "/home/$candidate"
        end

        if run_pm2_candidate $candidate $CANDIDATE_HOME describe soog-backend >/dev/null 2>/dev/null
            set PM2_APP_USER $candidate
            set PM2_PREFIX $CANDIDATE_HOME
            break
        end

        if run_pm2_candidate $candidate $CANDIDATE_HOME describe soog-frontend >/dev/null 2>/dev/null
            set PM2_APP_USER $candidate
            set PM2_PREFIX $CANDIDATE_HOME
            break
        end
    end

    if test -n "$PM2_APP_USER"
        if run_pm2_candidate $PM2_APP_USER $PM2_PREFIX describe soog-backend >/dev/null 2>/dev/null
            echo "  - Restarting PM2 app ($PM2_APP_USER): soog-backend"
            run_pm2_candidate $PM2_APP_USER $PM2_PREFIX restart soog-backend --update-env; or exit 1
            set USED_PM2 1
        end

        if run_pm2_candidate $PM2_APP_USER $PM2_PREFIX describe soog-frontend >/dev/null 2>/dev/null
            echo "  - Restarting PM2 app ($PM2_APP_USER): soog-frontend"
            run_pm2_candidate $PM2_APP_USER $PM2_PREFIX restart soog-frontend --update-env; or exit 1
            set USED_PM2 1
        end
    end

    if test $USED_PM2 -eq 1
        echo "  - Saving PM2 process list for $PM2_APP_USER"
        run_pm2_candidate $PM2_APP_USER $PM2_PREFIX save; or exit 1
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
set -l BACKEND_OK 0
for _ in (seq 1 20)
    if curl -fsS http://127.0.0.1:10000/api/ollama/verify >/dev/null 2>/dev/null
        set BACKEND_OK 1
        break
    end
    sleep 2
end
if test $BACKEND_OK -ne 1
    echo "✖ Backend health check failed after retries"
    exit 1
end

set -l FRONTEND_OK 0
for _ in (seq 1 15)
    if curl -fsSI http://127.0.0.1:3000 >/dev/null 2>/dev/null
        set FRONTEND_OK 1
        break
    end
    sleep 2
end
if test $FRONTEND_OK -ne 1
    echo "✖ Frontend health check failed after retries"
    exit 1
end

echo "=== SOOG DEPLOY DONE ==="
