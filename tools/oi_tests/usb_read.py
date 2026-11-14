import serial, time, struct
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
time.sleep(0.1)
s.reset_input_buffer()
s.write(bytes([128])); s.flush(); time.sleep(0.2)
s.reset_input_buffer(); s.write(bytes([142, 35])); r = s.read(1)
print("OI mode:", {0:'off',1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else "no reply")
for n in range(6):
    s.reset_input_buffer(); s.write(bytes([142, 3])); d = s.read(10)
    if len(d) == 10:
        chg, mv, ma, t, q, cap = struct.unpack('>BHhbHH', d)
        print(f"{n+1}: {d.hex()}  state={chg} {mv/1000:.2f}V {ma}mA {t}C {q}/{cap}mAh", flush=True)
    else:
        print(f"{n+1}: {len(d)} bytes {d.hex()}", flush=True)
    time.sleep(0.3)
s.close()