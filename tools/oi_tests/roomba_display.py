import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
s.write(bytes([128])); s.flush(); time.sleep(0.15)
s.write(bytes([131])); s.flush(); time.sleep(0.15)
s.write(bytes([164]) + b'HELO'); s.flush(); time.sleep(0.05)      # digit LEDs ASCII
s.write(bytes([139, 0x0F, 128, 255])); s.flush()                     # all LEDs on, power LED orange, full
print(time.strftime('%H:%M:%S'), "sent: 128, 131, 164 'HELO', 139 LEDs")
time.sleep(1)
s.close()