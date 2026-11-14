import serial, time, struct
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
def mode():
    s.reset_input_buffer(); w(142, 35); r = s.read(1)
    return {0:'off',1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else 'no reply'
def log(m): print(time.strftime('%H:%M:%S'), m, flush=True)
def stop():
    for _ in range(3):
        w(137, 0, 0, 0, 0); time.sleep(0.05)
w(128); time.sleep(0.2); w(131); time.sleep(0.2)
m = mode(); log(f"mode: {m}")
if m != 'safe':
    log("not in safe mode -> NOT driving"); stop(); s.close(); raise SystemExit(1)
vel = struct.pack('>h', 100); rad = struct.pack('>H', 0x8000)   # 100 mm/s, straight
try:
    w(137, *vel, *rad); log("drive forward 100 mm/s")
    time.sleep(1.0)
finally:
    stop(); log("stop x3")
time.sleep(0.2)
log(f"mode after: {mode()}")
w(128); time.sleep(0.1); log(f"back to passive: {mode()}")
s.close()