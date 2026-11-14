import rclpy, time, math, numpy as np, cv2
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid
from tf2_ros import Buffer, TransformListener
from roomba_perception.explorer import frontier_goals, find_frontiers
rclpy.init(); n=Node('goal_probe'); tb=Buffer(); TransformListener(tb,n); d={}
n.create_subscription(OccupancyGrid,'/global_costmap/costmap',lambda m: d.setdefault('g',m),QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))
t0=time.time()
while time.time()-t0<8 and ('g' not in d or not tb.can_transform('map','base_link',Time())): rclpy.spin_once(n,timeout_sec=0.1)
m=d['g']; i=m.info; g=np.array(m.data,np.int16).reshape(i.height,i.width); res=i.resolution; ox,oy=i.origin.position.x,i.origin.position.y
t=tb.lookup_transform('map','base_link',Time()).transform.translation
rc=(int((t.y-oy)/res), int((t.x-ox)/res))
t1=time.perf_counter(); goals, seed = frontier_goals(g, rc, 40, 98, 15, 10, 0.3/res, 0.8/res); dt=(time.perf_counter()-t1)*1000
print('robot', (round(t.x,2), round(t.y,2)), 'seed', seed, 'raw frontier groups', len(find_frontiers(g,40,10)), 'goals', len(goals), '%.0f ms' % dt)
for (gy,gx),(cy,cx),size in sorted(goals, key=lambda q: math.hypot(q[0][0]-rc[0], q[0][1]-rc[1])):
    print('  goal (%.2f, %.2f)  frontier (%.2f, %.2f)  size %d  dist %.1f m' % (ox+(gx+.5)*res, oy+(gy+.5)*res, ox+(cx+.5)*res, oy+(cy+.5)*res, size, math.hypot(ox+(gx+.5)*res-t.x, oy+(gy+.5)*res-t.y)))
img=np.full((i.height,i.width,3),128,np.uint8); img[(g>=0)&(g<=40)]=(255,255,255); img[(g>40)&(g<99)]=(200,200,240); img[g>=99]=(0,0,0)
for cells,_ in find_frontiers(g,40,10): img[cells[:,0],cells[:,1]]=(0,140,255)
for (gy,gx),(cy,cx),_ in goals: cv2.line(img,(int(gx),int(gy)),(int(cx),int(cy)),(200,0,200),1); cv2.circle(img,(int(gx),int(gy)),2,(200,0,200),-1)
cv2.circle(img,(rc[1],rc[0]),3,(0,170,0),-1)
img=cv2.flip(img,0); k=max(1,int(700/max(i.width,i.height))); cv2.imwrite('/tmp/goals.png',cv2.resize(img,(i.width*k,i.height*k),interpolation=cv2.INTER_NEAREST))
