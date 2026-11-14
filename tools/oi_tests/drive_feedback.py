import serial, time, struct
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.3)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
def q(pid, n, fmt):
    s.reset_input_buffer(); w(142, pid); d = s.read(n)
    return struct.unpack('>' + fmt, d)[0] if len(d) == n else None
def mode():
    m = q(35, 1, 'B'); return {1:'passive',2:'safe',3:'full'}.get(m, m)
def stop():
    for _ in range(3): w(137, 0, 0, 0, 0); time.sleep(0.05)
def sample(t0, tag):
    L, R = q(43, 2, 'H'), q(44, 2, 'H'); iL, iR = q(54, 2, 'h'), q(55, 2, 'h'); st = q(58, 1, 'B')
    print(f"{time.time()-t0:5.2f}s {tag:5s} encL={L} encR={R}  curL={iL}mA curR={iR}mA  stasis={st}", flush=True)
    return L, R
w(128); time.sleep(0.2); w(131); time.sleep(0.2)
m = mode(); print("mode:", m)
if m != 'safe':
    print("not safe -> not driving"); stop(); s.close(); raise SystemExit(1)
q(19, 2, 'h'); q(20, 2, 'h')   # reset distance/angle accumulators
t0 = time.time()
L0, R0 = sample(t0, "start")
try:
    w(137, *struct.pack('>h', 100), 0x80, 0x00)
    while time.time() - t0 < 1.1:
        sample(t0, "drive")
finally:
    stop()
time.sleep(0.3)
L1, R1 = sample(t0, "stop")
d = q(19, 2, 'h'); a = q(20, 2, 'h')
if None not in (L0, R0, L1, R1):
    dL = (L1 - L0) % 65536; dR = (R1 - R0) % 65536
    mm = lambda c: c * 72.0 * 3.14159 / 508.8
    print(f"\nencoder delta L={dL} R={dR} counts  ->  ~{mm(dL):.0f} mm / ~{mm(dR):.0f} mm")
print(f"odometry: distance={d} mm  angle={a} deg")
w(128); time.sleep(0.1); print("back to:", mode())
s.close()