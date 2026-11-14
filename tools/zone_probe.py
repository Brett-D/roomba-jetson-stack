import rclpy, numpy as np, time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
rclpy.init(); n = Node('zone_probe'); buf = Buffer(); TransformListener(buf, n)
clouds = []
n.create_subscription(PointCloud2, '/perception/obstacles', lambda m: clouds.append(m), 5)
t0 = time.time()
while len(clouds) < 6 and time.time() - t0 < 8: rclpy.spin_once(n, timeout_sec=0.1)
for m in clouds[-4:]:
    pts = np.frombuffer(m.data, np.float32).reshape(-1, 3)
    tf = buf.lookup_transform('base_link', m.header.frame_id, Time())
    q = tf.transform.rotation; tr = tf.transform.translation
    x, y, z, w = q.x, q.y, q.z, q.w
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    b = pts @ R.T + np.array([tr.x, tr.y, tr.z])
    stop = b[(b[:, 0] > 0.17) & (b[:, 0] < 0.55) & (np.abs(b[:, 1]) < 0.19)]
    slow = b[(b[:, 0] > 0.17) & (b[:, 0] < 0.85) & (np.abs(b[:, 1]) < 0.24)]
    msg = f'total {len(b):5d}  stop zone {len(stop):4d}  slowdown zone {len(slow):4d}'
    if len(slow):
        msg += f'  nearest x {slow[:,0].min():.2f} m, heights {np.percentile(slow[:,2],10):.2f}-{np.percentile(slow[:,2],90):.2f} m'
    print(msg)