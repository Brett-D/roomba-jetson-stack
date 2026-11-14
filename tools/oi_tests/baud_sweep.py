import serial, time, sys
bauds = [19200, 115200, 57600, 38400, 28800, 14400, 9600, 4800, 2400]
start = int(sys.argv[1]) if len(sys.argv) > 1 else 1
results = []
for i, baud in enumerate(bauds, 1):
    if i < start: continue
    try:
        s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.3)
    except Exception as e:
        print(f"step {i:2d}: {baud:6d} baud  -> could not open port: {e}", flush=True); continue
    s.reset_input_buffer()
    bt = 10.0 / baud
    s.write(bytes([128])); s.flush(); time.sleep(0.1 + bt)
    s.write(bytes([131])); s.flush(); time.sleep(0.1 + bt)
    notes = []
    for k in range(i): notes += [84 if k % 2 == 0 else 76, 10]
    s.write(bytes([140, 0, i] + notes)); s.flush(); time.sleep(0.05 + 30 * bt)
    s.write(bytes([141, 0])); s.flush(); time.sleep(0.2)
    s.reset_input_buffer()
    s.write(bytes([142, 35])); s.flush()
    r = s.read(4)
    print(f"step {i:2d}: {baud:6d} baud  -> {i} beep(s)  reply={r.hex() or 'none'}", flush=True)
    results.append((baud, r))
    time.sleep(i * 0.17 + 2.0)
    s.write(bytes([128])); s.flush(); time.sleep(0.05)
    s.close()
print("valid OI-mode replies at:", [b for b, r in results if len(r) == 1 and r[0] in (1, 2, 3)] or "none")