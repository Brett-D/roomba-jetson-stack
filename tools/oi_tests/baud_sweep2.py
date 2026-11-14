import serial, time, sys
bauds = [int(b) for b in sys.argv[1:]] or [76800, 230400, 56000, 128000, 250000, 460800, 7200, 3600, 921600]
for i, baud in enumerate(bauds, 1):
    try:
        s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.3)
    except Exception as e:
        print(f"step {i:2d}: {baud:6d} baud  -> could not open: {e}", flush=True); continue
    bt = 10.0 / baud
    for op in (128, 130, 131):
        s.write(bytes([op])); s.flush(); time.sleep(0.1 + bt)
    notes = []
    for k in range(i): notes += [84 if k % 2 == 0 else 76, 10]
    s.write(bytes([140, 0, i] + notes)); s.flush(); time.sleep(0.05 + 30 * bt)
    s.write(bytes([141, 0])); s.flush(); time.sleep(0.2)
    s.reset_input_buffer()
    s.write(bytes([142, 35])); s.flush()
    r = s.read(4)
    print(f"step {i:2d}: {baud:6d} baud  -> {i} beep(s)  reply={r.hex() or 'none'}", flush=True)
    time.sleep(i * 0.17 + 2.5)
    s.write(bytes([128])); s.flush(); time.sleep(0.05)
    s.close()