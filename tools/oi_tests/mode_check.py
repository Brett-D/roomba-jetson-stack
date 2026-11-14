import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
def q(pkt, n):
    s.reset_input_buffer(); w(142, pkt); return s.read(n)
modes = {0:'off',1:'passive',2:'safe',3:'full'}
w(128); time.sleep(0.2)
r = q(35, 1); print("after 128 START   -> OI mode pkt35:", r.hex() or 'none', modes.get(r[0], '?') if r else '')
p1 = q(1, 10); print("sensor group 1 (bumps/wheeldrops, wall, cliffs, vwall, overcurrent, dirt):", p1.hex())
if p1:
    b = p1[0]; print(f"   bumps/wheeldrops byte = {b:08b}  (bits4..2 wheeldrop caster/left/right, bits1..0 bump L/R)")
    if len(p1) >= 6: print(f"   cliffs L/FL/FR/R = {list(p1[2:6])}")
w(130); time.sleep(0.2)
r = q(35, 1); print("after 130 CONTROL -> OI mode:", r.hex() or 'none', modes.get(r[0], '?') if r else '')
w(131); time.sleep(0.2)
r = q(35, 1); print("after 131 SAFE    -> OI mode:", r.hex() or 'none', modes.get(r[0], '?') if r else '')
w(132); time.sleep(0.2)
r = q(35, 1); print("after 132 FULL    -> OI mode:", r.hex() or 'none', modes.get(r[0], '?') if r else '')
w(128); s.close()