import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
for _ in range(3):
    w(128); time.sleep(0.1); w(131); time.sleep(0.1)
    w(137, 0, 0, 0, 0); w(138, 0); time.sleep(0.1)
s.reset_input_buffer(); w(142, 35); r = s.read(1)
print(time.strftime('%H:%M:%S'), "STOP sent (safe mode, drive 0, motors off); mode:", {1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else 'no reply')
s.close()