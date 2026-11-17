#!/usr/bin/env python3
"""Stand-in battery_learner for the control page's ROS test (ros_controls.sh), run in a private
ROS_DOMAIN_ID: the /battery_learner/pack_swapped service (std_srvs/Trigger). Each call starts the
next pack number, written to $FAKE_DIR/pack (fake_driver.py puts it in battery_state's
serial_number), logs to $FAKE_DIR/driver.jsonl and answers "new pack N at 100 %".
  fake_battery_learner.py <token>      (the token is only there for the page's --learner-match)"""
import json
import os
import time

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

DIR = os.environ['FAKE_DIR']


class FakeLearner(Node):
    def __init__(self):
        super().__init__('battery_learner')
        self.create_service(Trigger, '~/pack_swapped', self.on_swap)

    def on_swap(self, req, res):
        path = os.path.join(DIR, 'pack')
        try:
            with open(path) as f:
                pack = int(f.read().strip() or 0) + 1
        except (OSError, ValueError):
            pack = 1
        with open(path, 'w') as f:
            f.write(str(pack))
        with open(os.path.join(DIR, 'driver.jsonl'), 'a') as f:
            f.write(json.dumps({'t': time.time(), 'kind': 'pack_swapped', 'pack': pack}) + '\n')
        res.success = True
        res.message = f'new pack {pack} at 100 %'
        return res


def main():
    if os.environ.get('ROS_DOMAIN_ID', '0') in ('', '0'):
        raise SystemExit('fake_battery_learner.py: set a private ROS_DOMAIN_ID (domain 0 is the robot)')
    rclpy.init()
    try:
        rclpy.spin(FakeLearner())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
