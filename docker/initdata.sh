#!/bin/bash
set -e  # Stop script on error

echo "Creating dataset directory..."
mkdir -p ~/data/medquad && cd ~/data/medquad

echo "Installing rclone..."
curl https://rclone.org/install.sh | sudo bash

echo "Enabling 'allow_other' in /etc/fuse.conf..."
sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

echo "Creating rclone config directory..."
mkdir -p ~/.config/rclone

echo "Writing rclone configuration..."
cat <<EOF > ~/.config/rclone/rclone.conf
[chi_tacc]
type = swift
user_id = $YOUR_USER_ID
application_credential_id = $APP_CRED_ID
application_credential_secret = $APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

echo "📂 Checking remote containers:"
rclone lsd chi_tacc: || { echo "Failed to list remote containers. Check credentials."; exit 1; }

export RCLONE_CONTAINER=object-persist-project17

echo "Mounting container to /mnt/object..."
sudo mkdir -p /home/cc/mnt/object
sudo chown -R cc /home/cc/mnt/object
sudo chgrp -R cc /home/cc/mnt/object

rclone mount chi_tacc:$RCLONE_CONTAINER /home/cc/mnt/object --allow-other --daemon


echo "📁 Contents of /mnt/object:"
ls /mnt/object
