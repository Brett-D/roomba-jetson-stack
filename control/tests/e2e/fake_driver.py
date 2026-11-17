#!/usr/bin/env python3
"""Stand-in roomba_driver for the control page's ROS test (ros_controls.sh), run in a private
ROS_DOMAIN_ID: the roomba/define_song service, the roomba/cleaning_motors subscription, roomba/mode
(latched) and roomba/sensors (5 Hz). The OI mode and a hazard come from the files mode and hazard
in $FAKE_DIR; every request is appended to $FAKE_DIR/driver.jsonl.
  fake_driver.py <token>      (the token is only there for the page's --driver-match)"""
import json
import os
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from roomba_msgs.msg import CleaningMotors, SensorState
from roomba_msgs.srv import DefineSong
from std_msgs.msg import String

DIR = os.environ['FAKE_DIR']
MODES = {'off': 0, 'passive': 1, 'safe': 2, 'full': 3}


def read(name, default):
    try:
        with open(os.path.join(DIR, name)) as f:
            return f.read().strip() or default
    except OSError:
        return default


def log(kind, **fields):
    with open(os.path.join(DIR, 'driver.jsonl'), 'a') as f:
        f.write(json.dumps({'t': time.time(), 'kind': kind, **fields}) + '\n')


class FakeDriver(Node):
    def __init__(self):
        super().__init__('roomba_driver')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_mode = self.create_publisher(String, 'roomba/mode', latched)
        self.pub_sensors = self.create_publisher(SensorState, 'roomba/sensors', 2)
        self.create_subscription(CleaningMotors, 'roomba/cleaning_motors', self.on_motors, 10)
        self.create_service(DefineSong, 'roomba/define_song', self.on_song)
        self.mode = None
        self.create_timer(0.2, self.tick)

    def tick(self):
        mode = read('mode', 'passive')
        if mode != self.mode:
            self.mode = mode
            self.pub_mode.publish(String(data=mode))
        msg = SensorState()
        msg.oi_mode = MODES.get(mode, 0)
        hazard = read('hazard', '')
        if hazard:
            setattr(msg, hazard, True)
        self.pub_sensors.publish(msg)

    def on_motors(self, m):
        log('motors', main_brush=m.main_brush, side_brush=m.side_brush, vacuum=m.vacuum, use_pwm=m.use_pwm)

    def on_song(self, req, res):
        log('song', song=req.song_number, notes=list(req.notes), durations=list(req.durations), play=req.play)
        what = f'song {req.song_number} defined and played'
        res.success = not req.play or self.mode in ('safe', 'full')
        res.message = f'{what} sent' if res.success else f'{what} needs Safe or Full mode; robot is in {self.mode} mode'
        return res


def main():
    if os.environ.get('ROS_DOMAIN_ID', '0') in ('', '0'):
        raise SystemExit('fake_driver.py: set a private ROS_DOMAIN_ID (domain 0 is the robot)')
    rclpy.init()
    try:
        rclpy.spin(FakeDriver())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
