import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
def mode():
    s.reset_input_buffer(); w(142, 35); r = s.read(1)
    return {0:'off',1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else 'no reply'
w(128); time.sleep(0.2); w(131); time.sleep(0.2)
m = mode()
if m not in ('safe', 'full'):
    w(132); time.sleep(0.2); m = mode()
w(164, *b'    '); time.sleep(0.05); w(163, 0, 0, 0, 0); time.sleep(0.3)
w(128); time.sleep(0.1)
print(time.strftime('%H:%M:%S'), f"display cleared (was in {m}); now: {mode()}")
s.close()