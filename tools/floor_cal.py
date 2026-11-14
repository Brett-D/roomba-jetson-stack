import rclpy, time, json, statistics
from rclpy.node import Node
from std_msgs.msg import String
rclpy.init(); n=Node('floor_cal'); H=[]; N=[0]
def cb(m):
    N[0]+=1; d=json.loads(m.data)
    if d.get('measured_height_m') is not None: H.append(d['measured_height_m'])
n.create_subscription(String,'/perception/floor_status',cb,10)
t0=time.time()
while time.time()-t0<25: rclpy.spin_once(n,timeout_sec=0.1)
print('frames', N[0], 'with floor', len(H))
if H: print('median %.3f  min %.3f  max %.3f  stdev %.4f' % (statistics.median(H), min(H), max(H), statistics.pstdev(H)))
