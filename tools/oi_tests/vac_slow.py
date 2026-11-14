import serial, time
time.sleep(2)
for i, baud in enumerate([57600, 115200], 1):
    s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.6)
    def w(*bs):
        for b in bs:
            s.write(bytes([b])); s.flush(); time.sleep(0.01)
    w(128); time.sleep(0.1); w(130); time.sleep(0.1)
    w(138, 2)
    print(f"{time.strftime('%H:%M:%S')}  #{i} {baud}: vacuum ON (slow bytes, 2 stop bits)", flush=True)
    time.sleep(2.5)
    w(138, 0); time.sleep(0.1)
    s.reset_input_buffer(); w(142, 3); d = s.read(10)
    print(f"           vacuum OFF; sensor reply {len(d)} bytes {d.hex()}", flush=True)
    w(128); s.close()
    time.sleep(5)