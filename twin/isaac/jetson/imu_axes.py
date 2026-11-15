import rosbag2_py, numpy as np
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu
r = rosbag2_py.SequentialReader()
r.open(rosbag2_py.StorageOptions(uri='/data/bags/robot_20261114_145937_fresh_map8', storage_id='mcap'), rosbag2_py.ConverterOptions('cdr', 'cdr'))
r.set_filter(rosbag2_py.StorageFilter(topics=['/camera/imu/data_raw']))
acc, gyr = [], []
while r.has_next() and len(acc) < 2000:
    t, d, ts = r.read_next()
    m = deserialize_message(d, Imu)
    acc.append((m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z))
    gyr.append((m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z))
a, g = np.array(acc), np.array(gyr)
print('IMU_REAL n', len(a), 'acc mean', a.mean(0).round(3), 'acc std', a.std(0).round(3), 'gyro mean', g.mean(0).round(4), 'gyro std', g.std(0).round(4), 'orientation_cov0', m.orientation_covariance[0])
