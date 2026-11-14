import serial, time, struct
for i, baud in enumerate([57600, 19200, 115200], 1):
    try:
        s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.5)
    except Exception as e:
        print(f"{i} beep(s) @ {baud}: could not open port: {e}", flush=True); continue
    s.reset_input_buffer()
    s.write(bytes([128])); s.flush(); time.sleep(0.15)   # START
    s.write(bytes([130])); s.flush(); time.sleep(0.15)   # CONTROL (SCI command mode -> safe)
    notes = []
    for k in range(i): notes += [84 if k % 2 == 0 else 76, 12]
    s.write(bytes([140, 0, i] + notes)); s.flush(); time.sleep(0.1)
    s.write(bytes([141, 0])); s.flush(); time.sleep(0.3)
    s.reset_input_buffer()
    s.write(bytes([142, 3])); s.flush()                  # sensor packet 3 (10 bytes, SCI + OI)
    d = s.read(10)
    print(f"{i} beep(s) @ {baud}: sent 128,130,song,play  reply={len(d)} bytes {d.hex()}", flush=True)
    if len(d) == 10:
        chg, mv, ma, t, q, cap = struct.unpack('>BHhbHH', d)
        print(f"   SUCCESS: {mv/1000:.2f} V, {ma} mA, {t} C, charge {q}/{cap} mAh")
    time.sleep(3)
    s.close()