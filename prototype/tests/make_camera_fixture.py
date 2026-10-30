"""Synthetic 72 BPM camera input for Playwright; not real-person validation."""
from pathlib import Path
import math
p=Path(__file__).parent/'synthetic-72bpm.y4m'
with p.open('wb') as f:
    f.write(b'YUV4MPEG2 W64 H64 F30:1 Ip A1:1 C420jpeg\n')
    for i in range(45*30):
        t=i/30;phase=2*math.pi*1.2*t;r=190+10*math.sin(phase)+3*math.sin(2*phase);g=30;b=20
        y=round(.299*r+.587*g+.114*b);u=round(-.168736*r-.331264*g+.5*b+128);v=round(.5*r-.418688*g-.081312*b+128)
        f.write(b'FRAME\n'+bytes([y])*4096+bytes([u])*1024+bytes([v])*1024)
print(p)
