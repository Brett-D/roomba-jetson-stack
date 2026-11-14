import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
s.write(bytes([128])); time.sleep(0.2); s.reset_input_buffer()
s.write(bytes([142, 35])); r = s.read(1)
print("mode:", r.hex() or "no reply (asleep?)")
s.close()