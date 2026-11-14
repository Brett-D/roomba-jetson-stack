import rclpy, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
rclpy.init(); n = Node('obs_probe'); buf = Buffer(); TransformListener(buf, n)
got = {}
n.create_subscription(PointCloud2, '/perception/obstacles', lambda m: got.setdefault('m', m), 2)
t0 = time.time()
while 'm' not in got and time.time() - t0 < 5: rclpy.spin_once(n, timeout_sec=0.1)
for _ in range(20): rclpy.spin_once(n, timeout_sec=0.05)
m = got['m']; pts = np.frombuffer(m.data, np.float32).reshape(-1, 3)
tf = buf.lookup_transform('base_link', m.header.frame_id, Time())
q = tf.transform.rotation; tr = tf.transform.translation
x, y, z, w = q.x, q.y, q.z, q.w
R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
b = pts @ R.T + np.array([tr.x, tr.y, tr.z])
print('obstacle points', len(b))
for lo, hi in ((0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 5)):
    s = b[(b[:, 0] >= lo) & (b[:, 0] < hi)]
    if len(s): print(f'x {lo}-{hi} m: {len(s):5d} pts, height above floor (base z) p10 {np.percentile(s[:,2],10):.2f} median {np.median(s[:,2]):.2f} p90 {np.percentile(s[:,2],90):.2f}, y range {s[:,1].min():.2f}..{s[:,1].max():.2f}')