import serial, time
bauds = [57600, 19200, 115200, 38400, 9600, 28800, 14400]
print("starting in 5 s...", flush=True); time.sleep(5)
for i, baud in enumerate(bauds, 1):
    try:
        s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.5)
    except Exception as e:
        print(f"step {i}: {baud} -> could not open: {e}", flush=True); continue
    s.write(bytes([128])); s.flush(); time.sleep(0.15)
    s.write(bytes([130])); s.flush(); time.sleep(0.15)
    notes = []
    for k in range(i): notes += [84 if k % 2 == 0 else 72, 20]
    s.write(bytes([140, 0, i] + notes)); s.flush(); time.sleep(0.1)
    s.write(bytes([141, 0])); s.flush()
    print(f"{time.strftime('%H:%M:%S')}  step {i}: {baud:6d} baud -> {i} beep(s)", flush=True)
    time.sleep(i * 0.32 + 4)
    s.close()