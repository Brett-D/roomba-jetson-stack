"""Live view of the Isaac Sim twin: shows the frames run_closed_loop.py writes (follow camera, robot camera inset,
HUD) in a window, with the current status line. Stays open across simulator restarts.

  python twin_viewer.py [--file E:\\twin\\run\\view.jpg]
"""
import argparse
import json
import os
import time
import tkinter as tk

from PIL import Image, ImageTk

ap = argparse.ArgumentParser()
ap.add_argument('--file', default=r'E:\twin\run\view.jpg')
ap.add_argument('--status', default=r'E:\twin\isaac_NOW.txt')
ap.add_argument('--state', default=r'E:\twin\run\isaac_state.json')
ap.add_argument('--scale', type=float, default=1.0)
args = ap.parse_args()

root = tk.Tk()
root.title('Isaac Sim twin - live (follow camera)')
root.configure(bg='black')
label = tk.Label(root, bg='black')
label.pack()
info = tk.Label(root, bg='black', fg='white', anchor='w', justify='left', font=('Segoe UI', 10), wraplength=1270)
info.pack(fill='x')
st = {'mtime': 0.0, 'img': None, 'seen': time.time()}


def tick():
    try:
        mt = os.path.getmtime(args.file)
        if mt != st['mtime']:
            im = Image.open(args.file)
            im.load()
            if args.scale != 1.0:
                im = im.resize((int(im.width * args.scale), int(im.height * args.scale)))
            st['img'] = ImageTk.PhotoImage(im)
            label.configure(image=st['img'])
            st['mtime'] = mt
            st['seen'] = time.time()
    except (OSError, ValueError, SyntaxError):
        pass
    if int(time.time() * 2) % 2 == 0:
        try:
            line = open(args.status, encoding='utf-8').read().strip()
        except OSError:
            line = ''
        try:
            s = json.load(open(args.state))
            line += (f"\nsim {s['t_sim']} s, real-time factor {s['rtf']}, robot ({s['x']}, {s['y']}), "
                     f"bumps {s['bumps']}, updated {s['wall_clock']}")
        except (OSError, ValueError, KeyError):
            pass
        if time.time() - st['seen'] > 10:
            line += '\n(no new frames for %.0f s - simulator starting, restarting or stopped)' % (time.time() - st['seen'])
        info.configure(text=line)
    root.after(40, tick)


root.after(100, tick)
root.mainloop()
