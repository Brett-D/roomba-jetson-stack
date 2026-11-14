import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.2)
s.reset_input_buffer()
s.write(bytes([128])); s.flush(); time.sleep(0.15)
s.write(bytes([7])); s.flush()
buf = b''; t = time.time()
while time.time() - t < 6:
    buf += s.read(256)
print(f"{time.strftime('%H:%M:%S')} reset sent; received {len(buf)} bytes")
print(repr(buf[:400]))
s.close()