import rclpy, time
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from nav_msgs.msg import Odometry
rclpy.init(); n=Node('lag'); L={'depth':[], 'imu':[], 'odom':[]}
def f(k): return lambda m: L[k].append(n.get_clock().now().nanoseconds*1e-9 - (m.header.stamp.sec+m.header.stamp.nanosec*1e-9))
n.create_subscription(Image,'/camera/depth/image_rect_raw',f('depth'),5)
n.create_subscription(Imu,'/camera/imu/data',f('imu'),50)
n.create_subscription(Odometry,'/odom',f('odom'),50)
t0=time.time()
while time.time()-t0<4: rclpy.spin_once(n,timeout_sec=0.05)
for k,v in L.items():
    v=sorted(v); print(k, len(v), 'lag median %.0f ms  max %.0f ms' % (1000*v[len(v)//2], 1000*v[-1]) if v else '')
