import serial, time
s = serial.Serial('/dev/ttyUSB0', 115200, timeout=0.5)
def w(*bs):
    for b in bs:
        s.write(bytes([b])); s.flush(); time.sleep(0.01)
def mode():
    s.reset_input_buffer(); w(142, 35); r = s.read(1)
    return {0:'off',1:'passive',2:'safe',3:'full'}.get(r[0], r.hex()) if r else 'no reply'
def log(m): print(time.strftime('%H:%M:%S'), m, flush=True)
def song(num, notes):
    flat = [x for nd in notes for x in nd]
    w(140, num, len(notes), *flat)
    return sum(d for _, d in notes) / 64.0
w(128); time.sleep(0.2); w(131); time.sleep(0.2)
log(f"mode: {mode()}")
w(164, *b'HELO'); log("display: HELO")
ode = [(76,16),(76,16),(77,16),(79,16),(79,16),(77,16),(76,16),(74,16),
       (72,16),(72,16),(74,16),(76,16),(76,24),(74,8),(74,32)]
t = song(0, ode); w(141, 0); log(f"playing Ode to Joy ({t:.1f} s)"); time.sleep(t + 0.5)
w(163, 0x7F, 0x7F, 0x7F, 0x7F); log("display: all segments")
scale = [(72,10),(74,10),(76,10),(77,10),(79,10),(81,10),(83,10),(84,24)]
t = song(1, scale); w(141, 1); log(f"playing C major scale ({t:.1f} s)"); time.sleep(t + 1.5)
log(f"mode at end: {mode()}")
for _ in range(2):
    w(128); time.sleep(0.1); w(131); time.sleep(0.1); w(137, 0, 0, 0, 0); w(138, 0); time.sleep(0.1)
log("stop sent")
s.close()