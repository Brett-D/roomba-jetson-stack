import serial, time, struct
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.6)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
w(128); time.sleep(0.1); w(130); time.sleep(0.1)
for n in range(6):
    s.reset_input_buffer(); w(142, 3); d = s.read(10)
    if len(d) == 10:
        chg, mv, ma, t, q, cap = struct.unpack('>BHhbHH', d)
        print(f"{n+1}: {d.hex()}  state={chg} {mv/1000:.2f}V {ma}mA {t}C {q}/{cap}mAh", flush=True)
    else:
        print(f"{n+1}: {len(d)} bytes {d.hex()}", flush=True)
    time.sleep(0.5)
w(128); s.close()