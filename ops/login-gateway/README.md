# 登录门

公网入口机上的回环登录服务 `admcube-login`：Nginx 用 `auth_request` 向它确认每个请求的会话，登录页与登出也由它提供，研究机上的控制台不参与鉴权。会话规则见[部署文档](../../docs/deployment-documentation.md#启动与检查-webui)。

本目录是登录门唯一的源：`gateway.py`、`login.css` 和 systemd 单元 `admcube-login.service`。登录页的 logo 与公安徽标就是控制台的 `src/autotrade/webui/static/logo.png` 和 `gongan.png`，安装时一并拷过去，这里不另存副本。测试在 `tests/unit/test_login_gateway.py`，不需要网络、入口机或 bcrypt 库。

口令和会话只在入口机上，永远不进仓库：

| 内容 | 入口机位置 |
| --- | --- |
| bcrypt 口令文件（`htpasswd -B` 生成） | `/etc/nginx/admcube-users`，`0640 root:www-data`，服务经附加组只读 |
| 会话存储（只存令牌的 SHA-256 摘要） | `/var/lib/admcube-login/sessions.json`，由单元的 `StateDirectory` 建成 0700 目录，文件 0600 |
| 登录审计（每次登录 POST 一行） | `/var/log/admcube-login/audit.jsonl` |

## 升级

先在研究机上跑 `pytest tests/unit/test_login_gateway.py`，再把 `gateway.py`、`login.css`、`admcube-login.service` 和上面两张图拷到入口机（ssh host `webui`）的同一个目录，在那里执行：

```bash
sudo install -o root -g root -m 0755 gateway.py /opt/admcube-login/gateway.py
sudo install -o root -g root -m 0644 login.css logo.png gongan.png /opt/admcube-login/
systemctl cat admcube-login | tail -n +2 | diff - admcube-login.service
sudo install -o root -g root -m 0644 admcube-login.service /etc/systemd/system/admcube-login.service
sudo systemctl daemon-reload
sudo systemctl restart admcube-login
```

`diff` 只为看清线上单元与仓库的差别：线上的 origin 等取值若与仓库不同，先把仓库改对再安装。只改登录门不需要 reload Nginx。装好后确认服务 active、重复打开登录页 CSRF 令牌不变、真实登录能进控制台、再 `sudo systemctl restart admcube-login` 一次会话仍然有效、登出后会话失效。

回滚是把 `/opt/admcube-login` 与单元文件恢复成旧版本，`daemon-reload` 后重启；程序和单元必须一起回滚，因为单元里的参数要与程序对应。

首次安装还要先建服务账户与目录：`sudo useradd --system --home /nonexistent --shell /usr/sbin/nologin admcube-login`、`sudo install -d -o root -g root -m 0755 /opt/admcube-login`、`sudo install -d -o admcube-login -g admcube-login -m 0750 /var/log/admcube-login`，装好文件后用 `sudo systemctl enable --now admcube-login` 启动；Nginx 站点由 `ops/nginx/install-admcubequant.sh` 安装。

## 让所有人重新登录

```bash
sudo systemctl stop admcube-login
sudo rm /var/lib/admcube-login/sessions.json
sudo systemctl start admcube-login
```

会话存储损坏时服务会拒绝启动，同样用这三步恢复。
