#!/usr/bin/env python3
"""Stand-in roomba_driver for the control page's ROS test (ros_controls.sh), run in a private
ROS_DOMAIN_ID: the roomba/define_song service, the roomba/cleaning_motors subscription, roomba/mode
(latched), roomba/sensors (5 Hz) and battery_state (1 Hz: 18.20 V, 56 %). The OI mode, a hazard,
battery_learner's pack number and learned capacity come from the files mode, hazard, pack and
capacity in $FAKE_DIR; every request is appended to $FAKE_DIR/driver.jsonl. While the file
motors_unplugged exists it has no roomba/cleaning_motors subscription (a publisher waits for one).
  fake_driver.py <token>      (the token is only there for the page's --driver-match)"""
import json
import os
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from roomba_msgs.msg import CleaningMotors, SensorState
from roomba_msgs.srv import DefineSong
from sensor_msgs.msg import BatteryState
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
        self.motors_sub = self.create_subscription(CleaningMotors, 'roomba/cleaning_motors', self.on_motors, 10)
        # Its own callback group: a slow answer (song_delay) doesn't hold up the rest.
        self.create_service(DefineSong, 'roomba/define_song', self.on_song, callback_group=MutuallyExclusiveCallbackGroup())
        self.pub_battery = self.create_publisher(BatteryState, 'battery_state', 10)
        self.mode = None
        self.create_timer(0.2, self.tick)
        self.create_timer(1.0, self.battery)

    def tick(self):
        unplugged = os.path.exists(os.path.join(DIR, 'motors_unplugged'))
        if unplugged and self.motors_sub is not None:
            self.destroy_subscription(self.motors_sub)
            self.motors_sub = None
        elif not unplugged and self.motors_sub is None:
            self.motors_sub = self.create_subscription(CleaningMotors, 'roomba/cleaning_motors', self.on_motors, 10)
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

    def battery(self):
        pack = read('pack', '')
        # battery_learner's format
        msg = BatteryState(voltage=18.2, percentage=0.56, serial_number=f'pack {pack}' if pack else '')
        try:
            msg.capacity = float(read('capacity', 'nan'))
        except ValueError:
            msg.capacity = float('nan')
        self.pub_battery.publish(msg)

    def on_motors(self, m):
        log('motors', main_brush=m.main_brush, side_brush=m.side_brush, vacuum=m.vacuum, use_pwm=m.use_pwm)

    def on_song(self, req, res):
        log('song', song=req.song_number, notes=list(req.notes), durations=list(req.durations), play=req.play)
        time.sleep(float(read('song_delay', '0')))
        what = f'song {req.song_number} defined and played'
        res.success = not req.play or self.mode in ('safe', 'full')
        res.message = f'{what} sent' if res.success else f'{what} needs Safe or Full mode; robot is in {self.mode} mode'
        return res


def main():
    if os.environ.get('ROS_DOMAIN_ID', '0') in ('', '0'):
        raise SystemExit('fake_driver.py: set a private ROS_DOMAIN_ID (domain 0 is the robot)')
    rclpy.init()
    try:
        executor = MultiThreadedExecutor()
        executor.add_node(FakeDriver())
        executor.spin()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
