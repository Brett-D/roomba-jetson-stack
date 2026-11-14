import serial, time, struct
try:
    s = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
except Exception as e:
    print("PORT OPEN FAILED:", e); raise SystemExit(1)
print("port open OK")
s.reset_input_buffer()
s.write(bytes([128])); s.flush(); time.sleep(0.15)
s.write(bytes([130])); s.flush(); time.sleep(0.15)
s.write(bytes([140, 1, 1, 79, 24])); s.write(bytes([141, 1])); s.flush(); time.sleep(0.6)
s.reset_input_buffer()
s.write(bytes([142, 3])); s.flush()
d = s.read(10)
print(f"packet 3 reply: {len(d)} bytes {d.hex()}")
if len(d) == 10:
    chg, mv, ma, t, q, cap = struct.unpack('>BHhbHH', d)
    states = ['not charging','reconditioning','full charging','trickle charging','waiting','charging fault']
    print(f"  charging : {states[chg] if chg < 6 else chg}")
    print(f"  voltage  : {mv/1000:.2f} V")
    print(f"  current  : {ma} mA")
    print(f"  temp     : {t} C")
    print(f"  charge   : {q} / {cap} mAh" + (f" ({100*q/cap:.0f}%)" if cap else ""))
s.reset_input_buffer()
s.write(bytes([142, 35])); s.flush()
m = s.read(1)
print("OI mode:", {0:'off',1:'passive',2:'safe',3:'full'}.get(m[0], m[0]) if m else "no reply")
s.write(bytes([128])); s.flush(); time.sleep(0.05)
s.close()