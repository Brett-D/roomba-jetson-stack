import rclpy, sys, time
from rclpy.node import Node
from roomba_msgs.msg import SensorState
F=['left','front_left','front_right','right']
dur=float(sys.argv[1]) if len(sys.argv)>1 else 20
rclpy.init(); n=Node('cliff_rec'); L=[]
n.create_subscription(SensorState,'/roomba/sensors',lambda m: L.append([getattr(m,f'cliff_{f}_signal') for f in F]+[sum(getattr(m,f'cliff_{f}') for f in F), m.wheel_drop_left or m.wheel_drop_right]),50)
t0=time.time()
while time.time()-t0<dur: rclpy.spin_once(n,timeout_sec=0.02)
print('samples', len(L), 'sensors', F, '+ cliff flags, wheel drop')
for i,f in enumerate(F): v=sorted(r[i] for r in L); print(f'{f:12s} min {v[0]:5d}  p10 {v[len(v)//10]:5d}  median {v[len(v)//2]:5d}  max {v[-1]:5d}')
print('frames with any cliff flag', sum(1 for r in L if r[4]), ' wheel drop', sum(1 for r in L if r[5]))
for r in L[::max(1,len(L)//20)]: print(r)
