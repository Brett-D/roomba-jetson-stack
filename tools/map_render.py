import rclpy, time, numpy as np, cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time
from rclpy.duration import Duration
rclpy.init(); n=Node('map_render'); tb=Buffer(); TransformListener(tb,n); d={}
tl=QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
for k,t,q in (('static','/rtabmap/map',tl),('obst','/global_costmap/obstacle_layer',10),('cost','/global_costmap/costmap',tl)):
    n.create_subscription(OccupancyGrid,t,(lambda k: lambda m: d.__setitem__(k,m))(k),q)
t0=time.time()
while time.time()-t0<8 and len(d)<3: rclpy.spin_once(n,timeout_sec=0.1)
print('got', list(d))
c=d['cost']; W,H,res=c.info.width,c.info.height,c.info.resolution; ox,oy=c.info.origin.position.x,c.info.origin.position.y
def grid(m):
    a=np.array(m.data,np.int16).reshape(m.info.height,m.info.width); out=np.full((H,W),-1,np.int16)
    dx=int(round((m.info.origin.position.x-ox)/res)); dy=int(round((m.info.origin.position.y-oy)/res))
    ys,xs=np.mgrid[0:a.shape[0],0:a.shape[1]]; X=xs+dx; Y=ys+dy; ok=(X>=0)&(X<W)&(Y>=0)&(Y<H)
    out[Y[ok],X[ok]]=a[ok]; return out
S=grid(d['static']); C=np.array(c.data,np.int16).reshape(H,W)
O=grid(d['obst']) if 'obst' in d else np.full((H,W),-1,np.int16)
img=np.full((H,W,3),128,np.uint8); img[S==0]=(255,255,255); img[S>=65]=(0,0,0)
img[(C>0)&(C<99)&(S!=100)]=(200,200,255)
img[(O>=99)&(S<65)]=(0,0,255)
img[(O>=99)&(S>=65)]=(0,0,128)
print('static occupied', int((S>=65).sum()), 'obstacle-layer lethal', int((O>=99).sum()), 'of which not in static map', int(((O>=99)&(S<65)).sum()))
t=tb.lookup_transform('map','base_link',Time(),timeout=Duration(seconds=2)).transform.translation
px,py=int((t.x-ox)/res),int((t.y-oy)/res); cv2.circle(img,(px,py),int(0.17/res),(0,160,0),1)
img=cv2.flip(img,0); k=max(1,int(800/max(W,H))); img=cv2.resize(img,(W*k,H*k),interpolation=cv2.INTER_NEAREST)
cv2.imwrite('/tmp/map_render.png',img)
