#!/usr/bin/env bash
# Launch Hermes Desktop (dev mode) against an already-running local backend.
# Override any of these via env, e.g. HERMES_DESKTOP_REMOTE_URL=127.0.0.1:9200 ./start-desktop-dev.sh
set -euo pipefail

cd "$(dirname "$0")"

export HERMES_DESKTOP_REMOTE_URL="${HERMES_DESKTOP_REMOTE_URL:-127.0.0.1:9119}"
export HERMES_DESKTOP_REMOTE_TOKEN="${HERMES_DESKTOP_REMOTE_TOKEN:-hermes-desktop-pycharm-debug-9119}"

host_port="${HERMES_DESKTOP_REMOTE_URL#*://}"
host_port="${host_port%%/*}"
host="${host_port%:*}"
port="${host_port##*:}"

if ! nc -z "$host" "$port" >/dev/null 2>&1; then
  echo "警告: 后端 $host_port 未响应，请确认 Hermes 后端已启动。" >&2
fi

if [ ! -x node_modules/.bin/concurrently ] || [ ! -d apps/desktop/node_modules/electron/dist ]; then
  echo "桌面端依赖不完整，正在执行 npm install --workspace apps/desktop ..."
  npm install --workspace apps/desktop
fi

echo "连接后端: $HERMES_DESKTOP_REMOTE_URL"
exec npm run dev --workspace apps/desktop
