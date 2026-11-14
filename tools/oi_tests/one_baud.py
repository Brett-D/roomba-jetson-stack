import serial, time, sys
baud = int(sys.argv[1])
s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.5)
s.write(bytes([128])); s.flush(); time.sleep(0.15)
s.write(bytes([130])); s.flush(); time.sleep(0.15)
s.write(bytes([140, 0, 4, 60, 24, 64, 24, 67, 24, 72, 48])); s.flush(); time.sleep(0.1)
for _ in range(2):
    s.write(bytes([141, 0])); s.flush(); time.sleep(2.5)
print(f"played rising tune twice at {baud} ({time.strftime('%H:%M:%S')})")
s.close()