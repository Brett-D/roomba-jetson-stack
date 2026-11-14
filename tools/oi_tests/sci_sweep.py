import serial, time, struct
bauds = [57600, 19200, 115200, 38400, 9600, 28800, 14400]
for i, baud in enumerate(bauds, 1):
    try:
        s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.5)
    except Exception as e:
        print(f"step {i}: {baud:6d} -> could not open port: {e}", flush=True); continue
    s.reset_input_buffer()
    s.write(bytes([128])); s.flush(); time.sleep(0.15)
    s.write(bytes([130])); s.flush(); time.sleep(0.15)
    notes = []
    for k in range(i): notes += [84 if k % 2 == 0 else 76, 12]
    s.write(bytes([140, 0, i] + notes)); s.flush(); time.sleep(0.1)
    s.write(bytes([141, 0])); s.flush(); time.sleep(0.3)
    s.reset_input_buffer()
    s.write(bytes([142, 3])); s.flush()
    d = s.read(10)
    print(f"step {i}: {baud:6d} -> {i} beep(s)  reply={len(d)} bytes {d.hex()}", flush=True)
    if len(d) == 10:
        chg, mv, ma, t, q, cap = struct.unpack('>BHhbHH', d)
        print(f"   SUCCESS: {mv/1000:.2f} V, {ma} mA, {t} C, charge {q}/{cap} mAh", flush=True)
    time.sleep(i * 0.2 + 2.5)
    s.close()