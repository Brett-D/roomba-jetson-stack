import serial, time
bauds = [57600, 19200, 115200, 38400, 9600]
print("starting in 3 s", flush=True); time.sleep(3)
for i, baud in enumerate(bauds, 1):
    s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.6)
    s.reset_input_buffer()
    s.write(bytes([128])); s.flush(); time.sleep(0.1)
    s.write(bytes([130])); s.flush(); time.sleep(0.1)
    s.write(bytes([138, 2])); s.flush()
    print(f"{time.strftime('%H:%M:%S')}  #{i} {baud:6d}: vacuum ON", flush=True)
    time.sleep(2)
    s.write(bytes([138, 0])); s.flush(); time.sleep(0.1)
    s.reset_input_buffer()
    s.write(bytes([142, 3])); s.flush()
    d = s.read(10)
    print(f"           vacuum OFF; sensor reply {len(d)} bytes {d.hex()}", flush=True)
    s.write(bytes([128])); s.flush(); time.sleep(0.05)
    s.close()
    time.sleep(5)