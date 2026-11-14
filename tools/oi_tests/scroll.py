import serial, time, sys
text = (sys.argv[1] if len(sys.argv) > 1 else "HELLO").upper()
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def mode():
    s.reset_input_buffer(); s.write(bytes([142, 35])); r = s.read(1)
    return {0:'off',1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else 'no reply'
def log(m): print(time.strftime('%H:%M:%S'), m, flush=True)
s.write(bytes([128])); time.sleep(0.2); s.write(bytes([131])); time.sleep(0.2)
m = mode(); log(f"mode: {m}")
if m not in ('safe', 'full'):
    s.write(bytes([132])); time.sleep(0.2); m = mode(); log(f"after FULL: {m}")
pad = "    " + text + "    "
log(f"scrolling '{text}'")
for _ in range(2):
    for i in range(len(pad) - 3):
        s.write(bytes([164]) + pad[i:i+4].encode('ascii')); s.flush()
        time.sleep(0.35)
s.write(bytes([164]) + b'    '); time.sleep(0.1)
s.write(bytes([128])); time.sleep(0.1)
log(f"cleared; mode now: {mode()}")
s.close()