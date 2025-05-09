#!/bin/bash
set -e

echo "Setting up rclone mount..."

mkdir -p /data/medquad
mkdir -p ~/.config/rclone

cat <<EOF > ~/.config/rclone/rclone.conf
[chi_tacc]
type = swift
user_id = ${YOUR_USER_ID}
application_credential_id = ${APP_CRED_ID}
application_credential_secret = ${APP_CRED_SECRET}
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
EOF

export RCLONE_CONTAINER=object-persist-project17

rclone mount chi_tacc:$RCLONE_CONTAINER /mnt/object --allow-other --daemon

ls /mnt/object
