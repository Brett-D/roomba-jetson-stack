import rclpy, time, json, numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from rclpy.duration import Duration
def R(q):
    x,y,z,w=q; return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
rclpy.init(); n=Node('obs_hist'); tb=Buffer(); TransformListener(tb,n)
clouds={'obs':[], 'floor':[]}; st=[]
def cb(k):
    def f(m):
        if len(clouds[k])<6: clouds[k].append(m)
    return f
n.create_subscription(PointCloud2,'/perception/obstacles',cb('obs'),5)
n.create_subscription(PointCloud2,'/perception/floor_points',cb('floor'),5)
n.create_subscription(String,'/perception/floor_status',lambda m: st.append(json.loads(m.data)['tilt_deg']),5)
t0=time.time()
while time.time()-t0<4: rclpy.spin_once(n,timeout_sec=0.1)
def tobase(m):
    t=tb.lookup_transform('base_link',m.header.frame_id,Time(),timeout=Duration(seconds=1)).transform
    p=np.array([[a,b,c] for a,b,c in pc2.read_points(m,('x','y','z'),skip_nans=True)],dtype=float)
    r=t.rotation; tr=t.translation
    return p@R((r.x,r.y,r.z,r.w)).T+np.array([tr.x,tr.y,tr.z])
F=np.vstack([tobase(m) for m in clouds['floor']]); O=np.vstack([tobase(m) for m in clouds['obs']])
fz=np.median(F[:,2]); print('frames', len(clouds['obs']), 'floor z in base_link (median)', round(fz,3), 'tilt from IMU', st[:5])
for lo,hi in ((0.3,0.7),(0.7,1.2),(1.2,2),(2,3),(3,4.5)):
    fs=F[(F[:,0]>=lo)&(F[:,0]<hi)]; os_=O[(O[:,0]>=lo)&(O[:,0]<hi)]
    hz=os_[:,2]-fz
    hist=[int(((hz>=a)&(hz<b)).sum()) for a,b in ((0.06,0.1),(0.1,0.2),(0.2,0.5),(0.5,2))]
    print(f'x {lo}-{hi} m: floor pts {len(fs):5d} (z med {np.median(fs[:,2]) if len(fs) else float("nan"):+.3f})  obstacle pts {len(os_):5d}  by height 6-10cm/10-20/20-50/50+: {hist}  |y|<0.25: {int((np.abs(os_[:,1])<0.25).sum())}')
