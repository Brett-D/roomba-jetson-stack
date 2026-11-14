import serial, time, struct, math, sys
PORT = '/dev/ttyUSB0'
VEL = int(sys.argv[1]) if len(sys.argv) > 1 else 100     # mm/s
DUR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   # s
MM_PER_COUNT = math.pi * 72.0 / 508.8                    # 500-series: 72 mm wheel, 508.8 counts/rev
s = serial.Serial(PORT, 115200, timeout=0.3)
IDS = [43, 44, 54, 55, 39]          # encL, encR, curL, curR, requested velocity
FMT = '>HHhhh'; N = struct.calcsize(FMT)
def snap():
    s.reset_input_buffer(); s.write(bytes([149, len(IDS)] + IDS)); d = s.read(N)
    return struct.unpack(FMT, d) if len(d) == N else None
def mode():
    s.reset_input_buffer(); s.write(bytes([142, 35])); r = s.read(1)
    return {0:'off',1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else 'no reply'
def stop():
    for _ in range(3): s.write(bytes([137, 0, 0, 0, 0])); time.sleep(0.03)
s.write(bytes([128])); time.sleep(0.2); s.write(bytes([131])); time.sleep(0.2)
m = mode(); print("mode:", m)
if m != 'safe':
    print("not in safe mode (wheels up? on dock?) -> not driving"); s.close(); raise SystemExit(1)
s.reset_input_buffer(); s.write(bytes([149, 2, 19, 20])); s.read(4)   # reset distance/angle accumulators
a = snap(); print(f"start    encL={a[0]:5d} encR={a[1]:5d}")
t0 = time.time(); samples = []
try:
    s.write(bytes([137]) + struct.pack('>hH', VEL, 0x8000))
    while time.time() - t0 < DUR:
        r = snap()
        if r: samples.append((time.time() - t0, *r))
finally:
    stop()
time.sleep(0.4)
b = snap()
for t, L, R, iL, iR, v in samples:
    print(f"{t:5.2f}s  encL={L:5d} encR={R:5d}  curL={iL:4d}mA curR={iR:4d}mA  vel_req={v}")
print(f"stop     encL={b[0]:5d} encR={b[1]:5d}  curL={b[2]}mA curR={b[3]}mA")
dL = (b[0] - a[0] + 32768) % 65536 - 32768
dR = (b[1] - a[1] + 32768) % 65536 - 32768
print(f"\nencoder delta: L={dL} counts ({dL*MM_PER_COUNT:.1f} mm)  R={dR} counts ({dR*MM_PER_COUNT:.1f} mm)")
s.reset_input_buffer(); s.write(bytes([149, 2, 19, 20])); d = s.read(4)
if len(d) == 4:
    dist, ang = struct.unpack('>hh', d); print(f"odometry: distance={dist} mm  angle={ang} deg")
print(f"expected ~{VEL*DUR:.0f} mm commanded ({len(samples)} samples during drive)")
s.write(bytes([128])); time.sleep(0.1); print("mode at end:", mode())
s.close()