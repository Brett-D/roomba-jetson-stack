import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
for _ in range(3):
    w(128); time.sleep(0.15)
s.reset_input_buffer(); w(142, 35); r = s.read(1)
print(time.strftime('%H:%M:%S'), "sent 128 x3 (back to passive). mode reply:", r.hex() if r else "none")
s.reset_input_buffer(); w(142, 3); d = s.read(10)
print("battery packet:", len(d), "bytes", d.hex())
s.close()