#!/bin/bash
set -e

echo "Creating dataset directory..."
mkdir -p /data/medquad && cd /data/medquad

curl https://rclone.org/install.sh | bash
sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf

mkdir -p ~/.config/rclone

cat <<EOF > ~/.config/rclone/rclone.conf
[chi_tacc]
type = swift
user_id = $YOUR_USER_ID
application_credential_id = $APP_CRED_ID
application_credential_secret = $APP_CRED_SECRET
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

rclone lsd chi_tacc:
mkdir -p /mnt/object
chown -R cc /mnt/object
chgrp -R cc /mnt/object
rclone mount chi_tacc:object-persist-project17 /mnt/object --allow-other --daemon
ls /mnt/object
