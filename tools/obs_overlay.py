import rclpy, time, numpy as np, cv2
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, CameraInfo, Image, CompressedImage
from sensor_msgs_py import point_cloud2 as pc2
rclpy.init(); n=Node('obs_overlay'); d={}
def keep(k): return lambda m: d.setdefault(k, m)
n.create_subscription(PointCloud2,'/perception/obstacles',keep('obs'),5)
n.create_subscription(PointCloud2,'/perception/floor_points',keep('floor'),5)
n.create_subscription(CameraInfo,'/camera/depth/camera_info',keep('info'),5)
n.create_subscription(Image,'/camera/left/image_rect_mono',keep('img'),5)
t0=time.time()
while time.time()-t0<5 and len(d)<4: rclpy.spin_once(n,timeout_sec=0.1)
print('got', list(d), 'obs frame', d['obs'].header.frame_id, 'img', d['img'].width, d['img'].height, d['img'].encoding, 'info', d['info'].width, d['info'].height)
im=np.frombuffer(d['img'].data,np.uint8).reshape(d['img'].height,d['img'].step)[:, :d['img'].width]
im=cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
P=d['info'].p; fx,fy,cx,cy=P[0],P[5],P[2],P[6]
sx=d['img'].width/d['info'].width; sy=d['img'].height/d['info'].height
for k,col in (('floor',(0,200,0)),('obs',(0,0,255))):
    p=np.array([[a,b,c] for a,b,c in pc2.read_points(d[k],('x','y','z'),skip_nans=True)])
    if not len(p): continue
    u=((p[:,0]*fx/p[:,2]+cx)*sx).astype(int); v=((p[:,1]*fy/p[:,2]+cy)*sy).astype(int)
    for a,b in zip(u,v):
        if 0<=a<im.shape[1] and 0<=b<im.shape[0]: cv2.circle(im,(a,b),2,col,-1)
cv2.imwrite('/tmp/obs_overlay.jpg',im)
