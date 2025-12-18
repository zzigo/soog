# on server


use ./deploy/deploy.fish  
or 
dsoog

also deploy.fish was added to .local/bin on server:

chmod +x /opt/soog/deploy/deploy.fish
mkdir -p ~/.local/bin
ln -s /opt/soog/deploy/deploy.fish ~/.local/bin/dsoog
echo $PATH | grep .local/bin
set -Ux fish_user_paths ~/.local/bin $fish_user_paths
exec fish


# on local

## Backend
source venv/bin/activate.fish
