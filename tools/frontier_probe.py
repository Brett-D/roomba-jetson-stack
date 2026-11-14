import rclpy, time, numpy as np, cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid
from roomba_perception.explorer import find_frontiers
rclpy.init(); n=Node('frontier_probe'); d={}
n.create_subscription(OccupancyGrid,'/global_costmap/costmap',lambda m: d.setdefault('g',m),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
t0=time.time()
while time.time()-t0<8 and 'g' not in d: rclpy.spin_once(n,timeout_sec=0.1)
m=d['g']; i=m.info; g=np.array(m.data,np.int16).reshape(i.height,i.width)
groups=find_frontiers(g,40,6)
print('grid', i.width, 'x', i.height, 'unknown', int((g<0).sum()), 'free', int(((g>=0)&(g<=40)).sum()), 'frontier groups', len(groups), 'sizes', sorted([s for _,s in groups], reverse=True)[:12])
img=np.full((i.height,i.width,3),128,np.uint8); img[(g>=0)&(g<=40)]=(255,255,255); img[g>40]=(160,160,220); img[g>=99]=(0,0,0)
for cells,_ in groups: img[cells[:,0],cells[:,1]]=(0,140,255)
img=cv2.flip(img,0); k=max(1,int(700/max(i.width,i.height))); cv2.imwrite('/tmp/frontiers.png',cv2.resize(img,(i.width*k,i.height*k),interpolation=cv2.INTER_NEAREST))
