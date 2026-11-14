import serial, time, struct
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.3)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
pk = [(14,1,'B','motor overcurrent bits'), (19,2,'h','distance mm (since last)'), (20,2,'h','angle deg (since last)'),
      (22,2,'H','battery voltage mV'), (23,2,'h','battery current mA'),
      (39,2,'h','requested velocity mm/s'), (43,2,'H','LEFT encoder counts'), (44,2,'H','RIGHT encoder counts'),
      (54,2,'h','left wheel motor current mA'), (55,2,'h','right wheel motor current mA'),
      (56,2,'h','main brush current mA'), (57,2,'h','side brush current mA'), (58,1,'B','stasis (caster moving)')]
w(128); time.sleep(0.2)
for pid, n, fmt, name in pk:
    s.reset_input_buffer(); w(142, pid); d = s.read(n)
    val = struct.unpack('>' + fmt, d)[0] if len(d) == n else None
    print(f"pkt {pid:2d} {name:30s}: {d.hex() or '-':6s} -> {val}", flush=True)
    time.sleep(0.05)
s.close()