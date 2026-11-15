#!/usr/bin/env python3
"""Runs `ros2 launch` so that SIGTERM and SIGINT stop it like Ctrl-C, also when a script started it
in the background (run_robot.sh, run_vslam.sh, ...):

  graceful.py ros2 launch <package> <launch file> [launch arguments]

Plain `ros2 launch` (launch 3.4, Jazzy) gets both signals wrong there:
- SIGTERM (`kill <pid>`, service managers): launch hears only signals that have a Python handler
  (it reads them from signal.set_wakeup_fd) and SIGTERM has none, so it dies on the spot and every
  node keeps running. Nodes still writing to its output pipes then die of SIGPIPE mid-shutdown
  (RTAB-Map without saving its map). Launch's own SIGTERM handler would not help: it only abandons
  the run ("can result in orphaned processes").
- SIGINT: a background job of a non-interactive script starts with SIGINT ignored and Python keeps
  it ignored, so launch never hears it (`kill -INT` does nothing; bash can't undo that either).
Here SIGINT gets Python's usual handler back and SIGTERM starts launch's normal shutdown: SIGINT to
every process, SIGTERM after sigterm_timeout and SIGKILL after sigkill_timeout more. Everything else
is plain `ros2 launch`, and the command line still contains "ros2 launch <package>" for pgrep/pkill.
"""
import signal
import sys

import launch
import launch.logging
from launch.utilities import signal_management
from ros2cli.cli import main as ros2


def main():
    if sys.argv[1:3] != ['ros2', 'launch']:
        sys.exit('usage: graceful.py ros2 launch <package> <launch file> [launch arguments]')
    services = []
    init = launch.LaunchService.__init__

    def remember(self, *args, **kwargs):
        init(self, *args, **kwargs)
        services.append(self)

    handle = signal_management.AsyncSafeSignalManager.handle

    def handle_without_sigterm(self, signum, handler):
        # Launch's own SIGTERM handler abandons the run and leaves the processes running.
        return handle(self, signum, None if signum == signal.SIGTERM else handler)

    stopping = []

    def stop():
        if not stopping:
            stopping.append(True)
            launch.logging.get_logger('launch').warning('SIGTERM: shutting down (SIGINT to every process)')
            for service in services:
                # A shutdown not caused by a terminal's Ctrl-C: launch sends each process SIGINT.
                service.shutdown(force_sync=True)

    def sigterm(signum, frame):
        loops = [s.event_loop for s in services if s.event_loop is not None]
        if not loops:
            sys.exit(128 + signum)   # launch isn't running (yet, or any more): nothing to stop
        for loop in loops:
            loop.call_soon_threadsafe(stop)

    launch.LaunchService.__init__ = remember
    signal_management.AsyncSafeSignalManager.handle = handle_without_sigterm
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, sigterm)
    sys.exit(ros2(argv=sys.argv[2:]))


if __name__ == '__main__':
    main()
