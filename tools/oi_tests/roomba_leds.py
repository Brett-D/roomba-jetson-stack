import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
s.write(bytes([128])); s.flush(); time.sleep(0.15)
s.write(bytes([130])); s.flush(); time.sleep(0.15)
s.write(bytes([140, 1, 1, 79, 12, 141, 1])); s.flush(); time.sleep(0.3)   # short beep = commands accepted
s.write(bytes([164]) + b'HELO'); s.flush(); time.sleep(0.05)
print(time.strftime('%H:%M:%S'), "beep + HELO sent; cycling LEDs for 6 s", flush=True)
for i in range(12):
    color = 0 if i % 2 == 0 else 255          # 0 = green, 255 = red
    s.write(bytes([139, 0x0F if i % 2 == 0 else 0x00, color, 255])); s.flush()
    time.sleep(0.5)
s.close()