import serial, time, sys
baud = int(sys.argv[1])
s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.5)
s.write(bytes([128])); s.flush(); time.sleep(0.15)
s.write(bytes([130])); s.flush(); time.sleep(0.15)
s.write(bytes([140, 1, 1, 79, 48])); s.flush(); time.sleep(0.1)
s.write(bytes([141, 1])); s.flush(); time.sleep(1.2)
print(f"{time.strftime('%H:%M:%S')}  single long beep sent at {baud}")
s.close()