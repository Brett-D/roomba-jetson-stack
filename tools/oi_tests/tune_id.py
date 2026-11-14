import serial, time
plan = [(115200, [72, 67, 64, 60], "FALLING"), (57600, [60, 64, 67, 72], "RISING")]
for baud, notes, name in plan:
    s = serial.Serial('/dev/ttyUSB0', baud, timeout=0.5)
    s.write(bytes([128])); s.flush(); time.sleep(0.15)
    s.write(bytes([130])); s.flush(); time.sleep(0.15)
    song = [140, 0, 4]
    for n in notes: song += [n, 24]
    s.write(bytes(song)); s.flush(); time.sleep(0.1)
    for _ in range(2):
        s.write(bytes([141, 0])); s.flush(); time.sleep(2.5)
    print(f"{time.strftime('%H:%M:%S')}  {baud}: played {name} tune twice", flush=True)
    s.close()
    time.sleep(5)