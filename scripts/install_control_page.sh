#!/usr/bin/env bash
# Installs the robot control page (control/) as the always-on service roomba-control:
# http://<jetson-host>.local/ (or http://<jetson-ip>/) from any browser, phone included.
#   scripts/install_control_page.sh       build (release), install, (re)start; run it again to update
# Run it as the robot's user (it uses sudo for the system parts). Environment:
#   ROBOT_WS      workspace whose scripts/run_robot.sh, stop_slam.sh and install/ the page uses
#                 (default: this repository)
#   CONTROL_PORT  port to serve on (80)
#   CARGO         cargo to build with (cargo on PATH, else ~/.cargo/bin/cargo)
# It installs:
#   /usr/local/bin/roomba-control                      the binary
#   /etc/systemd/system/roomba-control.service         User=<you>, starts at boot, Restart=on-failure
#   /etc/sudoers.d/roomba-control-poweroff             <you> may run exactly `systemctl poweroff`
# and enables lingering for the user (see below).
set -euo pipefail

if [ "$(id -u)" -eq 0 ]; then
    echo "Run this as the robot's user, not root (it uses sudo where needed)." >&2
    exit 1
fi
REPO=$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)
WS=$(readlink -f "${ROBOT_WS:-$REPO}")
PORT=${CONTROL_PORT:-80}
USER_NAME=$(id -un)
GROUP_NAME=$(id -gn)
CARGO=${CARGO:-$(command -v cargo || echo "$HOME/.cargo/bin/cargo")}
SERVICE=roomba-control
BIN=/usr/local/bin/roomba-control
UNIT=/etc/systemd/system/$SERVICE.service
SUDOERS=/etc/sudoers.d/roomba-control-poweroff

[ -x "$WS/scripts/run_robot.sh" ] || { echo "No $WS/scripts/run_robot.sh: set ROBOT_WS." >&2; exit 1; }

echo "== building $REPO/control (release)"
CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4} "$CARGO" build --release --locked --manifest-path "$REPO/control/Cargo.toml"
sudo install -m 755 "$REPO/control/target/release/roomba-control" "$BIN"

echo "== power-off permission: $SUDOERS"
# Only this one command, exactly as the page runs it (sudo -n /usr/bin/systemctl poweroff).
# The file name has no dot while it's checked, so sudo ignores the half-written file.
echo "$USER_NAME ALL=(root) NOPASSWD: /usr/bin/systemctl poweroff" | sudo tee "$SUDOERS.new" > /dev/null
sudo chmod 0440 "$SUDOERS.new"
sudo visudo -cqf "$SUDOERS.new"
sudo mv "$SUDOERS.new" "$SUDOERS"

echo "== lingering for $USER_NAME"
# The robot this service starts runs outside any login session. Without lingering, logind's
# RemoveIPC deletes the user's shared memory (Fast DDS's /dev/shm files) when the last SSH login
# ends, under the running robot. With it, the user counts as logged in from boot.
sudo loginctl enable-linger "$USER_NAME"

# The robot started from the page gets the resource limits of an SSH login (this shell's).
lim() { [ "$1" = unlimited ] && echo infinity || echo "$1$2"; }
NOFILE="$(lim "$(ulimit -Sn)"):$(lim "$(ulimit -Hn)")"
MEMLOCK="$(lim "$(ulimit -Sl)" K):$(lim "$(ulimit -Hl)" K)"
RTPRIO="$(lim "$(ulimit -Sr)"):$(lim "$(ulimit -Hr)")"

echo "== $UNIT"
sudo tee "$UNIT" > /dev/null << EOF
[Unit]
Description=Roomba control page: start / stop the robot, comms check, shut down (port $PORT)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=$USER_NAME
Group=$GROUP_NAME
# The robot it starts runs like one started from an SSH login: the user's groups (dialout for the
# Roomba's serial port, video / plugdev for the camera), HOME, a login shell (bash -l), the same
# limits and umask, and no device or file system sandboxing.
WorkingDirectory=~
EnvironmentFile=-/etc/environment
EnvironmentFile=-/etc/default/locale
UMask=$(umask)
LimitNOFILE=$NOFILE
LimitMEMLOCK=$MEMLOCK
LimitRTPRIO=$RTPRIO
TasksMax=infinity
ExecStart=$BIN --listen 0.0.0.0:$PORT --robot-ws "$WS"
# Port $PORT as a normal user (the page drops this for everything it starts).
AmbientCapabilities=CAP_NET_BIND_SERVICE
# The robot lives in this unit's cgroup: stopping or restarting the page must not stop it, and a
# robot node killed for lack of memory must not take the page down.
KillMode=process
OOMPolicy=continue
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE" > /dev/null
sudo systemctl restart "$SERVICE"
sleep 1
systemctl is-active --quiet "$SERVICE" || { systemctl status --no-pager "$SERVICE"; exit 1; }
echo "== $SERVICE is running: http://$(hostname).local/ or http://$(hostname -I | awk '{print $1}')/"
