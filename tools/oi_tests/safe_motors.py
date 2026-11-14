import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
def q(pkt, n):
    s.reset_input_buffer(); w(142, pkt); return s.read(n)
def mode():
    r = q(35, 1); return {0:'off',1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else 'no reply'
def log(m): print(time.strftime('%H:%M:%S'), m, flush=True)
w(128); time.sleep(0.2)
p1 = q(1, 10)
if p1:
    log(f"bumps/wheeldrops={p1[0]:05b} (wd caster/L/R, bump L/R)  cliffs L/FL/FR/R={list(p1[2:6])}")
w(131); time.sleep(0.2); m = mode(); log(f"after 131 SAFE: {m}")
if m != 'safe':
    w(130); time.sleep(0.2); m = mode(); log(f"after 130 CONTROL: {m}")
w(140, 2, 1, 84, 16); w(141, 2); time.sleep(0.6); log("marker beep")
w(138, 2); log("vacuum ON");     time.sleep(2.5); w(138, 0); time.sleep(1)
w(138, 1); log("side brush ON"); time.sleep(2.5); w(138, 0); time.sleep(1)
w(138, 4); log("main brush ON"); time.sleep(2.5); w(138, 0); time.sleep(1)
log(f"mode after motors: {mode()}")
w(128); s.close()