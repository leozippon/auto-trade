#!/bin/sh
# Install public vhosts on the Aliyun edge (ssh host: webui).
set -e
repo=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
remote=webui
ssh -o BatchMode=yes "$remote" 'sudo -n true; mkdir -p ~/admcube-nginx'
scp -o BatchMode=yes \
  "$repo/ops/nginx/aliyun/admcubequant-http.conf" \
  "$repo/ops/nginx/aliyun/admcubequant-https.conf" \
  "$repo/ops/nginx/aliyun/admcube-https.conf" \
  "$repo/ops/nginx/aliyun/admcube-http-limits.conf" \
  "$remote:~/admcube-nginx/"
ssh -o BatchMode=yes "$remote" 'set -e
sudo cp ~/admcube-nginx/admcubequant-http.conf /etc/nginx/sites-available/admcubequant-http
sudo cp ~/admcube-nginx/admcube-http-limits.conf /etc/nginx/conf.d/admcube-http-limits.conf
sudo ln -sfn /etc/nginx/sites-available/admcubequant-http /etc/nginx/sites-enabled/admcubequant-http
sudo nginx -t
sudo systemctl reload nginx
if ! sudo -n test -f /etc/letsencrypt/live/admcubequant.tj.cn/fullchain.pem; then
  sudo certbot certonly --webroot -w /var/lib/letsencrypt -d admcubequant.tj.cn \
    --non-interactive --agree-tos --register-unsafely-without-email
fi
sudo cp ~/admcube-nginx/admcubequant-https.conf /etc/nginx/sites-available/admcubequant-https
sudo ln -sfn /etc/nginx/sites-available/admcubequant-https /etc/nginx/sites-enabled/admcubequant-https
if ! sudo -n test -f /etc/letsencrypt/live/admcube-ip/fullchain.pem || ! sudo -n test -f /etc/letsencrypt/live/admcube-ip/privkey.pem; then
  echo "missing /etc/letsencrypt/live/admcube-ip/fullchain.pem or /etc/letsencrypt/live/admcube-ip/privkey.pem; refusing to enable IP vhost" >&2
  exit 1
fi
sudo cp ~/admcube-nginx/admcube-https.conf /etc/nginx/sites-available/admcube-https
sudo ln -sfn /etc/nginx/sites-available/admcube-https /etc/nginx/sites-enabled/admcube-https
sudo nginx -t
sudo systemctl reload nginx
echo ok
'
