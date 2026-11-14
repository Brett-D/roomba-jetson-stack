import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
for op in (128, 130, 132):
    s.write(bytes([op])); s.flush(); time.sleep(0.15)
s.write(bytes([140, 1, 2, 79, 10, 84, 10, 141, 1])); s.flush(); time.sleep(0.4)   # two-tone = full mode reached
s.write(bytes([163, 0x7F, 0x7F, 0x7F, 0x7F])); s.flush()                          # all display segments on
s.write(bytes([162, 0xFF, 0xFF])); s.flush()                                      # scheduling LEDs on
print(time.strftime('%H:%M:%S'), "full mode; two-tone; blinking LEDs 8 s", flush=True)
for i in range(16):
    s.write(bytes([139, 0xFF if i % 2 == 0 else 0x00, 0 if i % 2 == 0 else 255, 255])); s.flush()
    time.sleep(0.5)
s.write(bytes([128])); s.flush(); time.sleep(0.05)   # back to passive (safe to leave)
s.close()