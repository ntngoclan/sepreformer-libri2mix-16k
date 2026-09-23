"""Vector architecture figures, based on audit_ltrr_shapes.py and actual sources.

Dependencies: reportlab, pymupdf. Pass --extra-pythonpath for isolated installs.
Run the shape audit first. This generator refuses stale source fingerprints.
"""
import argparse
import hashlib
import html
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
INK = "#24344B"
BLUE = "#266489"
ORANGE = "#B57626"
GREEN = "#398260"
PURPLE = "#785095"
GRAY = "#69788B"
PALE = {BLUE: "#F1F7FC", ORANGE: "#FFFAF1", GREEN: "#F2F9F4", PURPLE: "#F8F3FB", GRAY: "#F6F8FA"}


class Figure:
    def __init__(self, name, width, height, title, subtitle):
        from reportlab.pdfgen import canvas
        self.name, self.w, self.h = name, width, height
        self.pdf = canvas.Canvas(str(OUT / (name + ".pdf")), pagesize=(width, height))
        self.pdf.setTitle(title)
        self.pdf.setAuthor("SepReformer project / source-verified architecture")
        self.svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                    '<rect width="100%" height="100%" fill="white"/>',
                    f'<title>{html.escape(title)}</title>', f'<desc>{html.escape(subtitle)}</desc>']
        self.text(45, 58, title, 40, bold=True, anchor="start")
        self.text(45, 106, subtitle, 23, color=GRAY, anchor="start")
        self.path([(45, 132), (width-45, 132)], color="#CBD6E1", arrow=False)

    def text(self, x, y, value, size=24, color=INK, bold=False, anchor="middle"):
        from reportlab.lib.colors import HexColor
        from reportlab.pdfbase import pdfmetrics
        font = "DiagramBold" if bold else "Diagram"
        for index, line in enumerate(str(value).split("\n")):
            baseline = y + index * size * 1.28
            tw = pdfmetrics.stringWidth(line, font, size)
            left = x - (tw/2 if anchor == "middle" else tw if anchor == "end" else 0)
            assert 0 <= left and left+tw <= self.w and size <= baseline <= self.h, (self.name, "text outside canvas", line)
            missing = [char for char in line if ord(char) not in pdfmetrics.getFont(font).face.charToGlyph]
            assert not missing, ("Missing font glyph", missing)
            self.pdf.setFillColor(HexColor(color))
            self.pdf.setFont(font, size)
            draw = {"middle": self.pdf.drawCentredString, "start": self.pdf.drawString, "end": self.pdf.drawRightString}[anchor]
            draw(x, self.h-baseline, line)
            self.svg.append(f'<text x="{x}" y="{baseline}" text-anchor="{anchor}" font-family="Arial, DejaVu Sans, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}">{html.escape(line)}</text>')

    def rect(self, x, y, w, h, color=BLUE, fill="#FFFFFF", dashed=False, radius=12):
        from reportlab.lib.colors import HexColor
        self.pdf.setStrokeColor(HexColor(color)); self.pdf.setFillColor(HexColor(fill))
        self.pdf.setLineWidth(2.3); self.pdf.setDash([9, 6] if dashed else [])
        self.pdf.roundRect(x, self.h-y-h, w, h, radius, stroke=1, fill=1)
        self.pdf.setDash([])
        dash = ' stroke-dasharray="9 6"' if dashed else ''
        self.svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" stroke="{color}" stroke-width="2.3" fill="{fill}"{dash}/>')

    def box(self, x, y, w, h, title, detail="", color=BLUE, size=24):
        from reportlab.pdfbase.pdfmetrics import stringWidth
        lines = title.split("\n") + (detail.split("\n") if detail else [])
        for line in lines:
            assert stringWidth(line, "DiagramBold", size) < w-12, (self.name, line, w)
        self.rect(x, y, w, h, color)
        title_lines = title.split("\n")
        total = len(title_lines) + (len(detail.split("\n")) if detail else 0)
        start = y + h/2 - (total-1)*size*1.28/2 + size*.32
        self.text(x+w/2, start, title, size, color, True)
        if detail:
            self.text(x+w/2, start+len(title_lines)*size*1.28, detail, size, INK)

    def panel(self, x, y, w, h, title, note, color):
        self.rect(x, y, w, h, color, PALE[color], dashed=True, radius=18)
        self.text(x+w/2, y+42, title, 29, color, True)
        self.text(x+w/2, y+77, note, 21, color)

    def path(self, points, color=INK, dashed=False, arrow=True, width=2.7):
        from reportlab.lib.colors import HexColor
        self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(width)
        self.pdf.setDash([9, 6] if dashed else [])
        p = self.pdf.beginPath(); p.moveTo(points[0][0], self.h-points[0][1])
        for x, y in points[1:]: p.lineTo(x, self.h-y)
        self.pdf.drawPath(p); self.pdf.setDash([])
        dash = ' stroke-dasharray="9 6"' if dashed else ''
        coords = " ".join(f"{x},{y}" for x, y in points)
        self.svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"{dash}/>')
        if arrow:
            x, y = points[-1]; px, py = points[-2]
            a = math.atan2(y-py, x-px); length, half = 13, 5.5
            tri = [(x,y), (x-length*math.cos(a)+half*math.sin(a),y-length*math.sin(a)-half*math.cos(a)),
                   (x-length*math.cos(a)-half*math.sin(a),y-length*math.sin(a)+half*math.cos(a))]
            p=self.pdf.beginPath(); p.moveTo(tri[0][0],self.h-tri[0][1])
            for tx,ty in tri[1:]: p.lineTo(tx,self.h-ty)
            p.close(); self.pdf.setFillColor(HexColor(color)); self.pdf.drawPath(p,stroke=0,fill=1)
            self.svg.append(f'<polygon points="{" ".join(f"{tx},{ty}" for tx,ty in tri)}" fill="{color}"/>')

    def dot(self, x, y, color=INK):
        from reportlab.lib.colors import HexColor
        self.pdf.setFillColor(HexColor(color)); self.pdf.circle(x,self.h-y,4,stroke=0,fill=1)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')

    def op(self, x, y, label, color=PURPLE, r=24):
        from reportlab.lib.colors import HexColor
        self.pdf.setFillColor(HexColor("#FFFFFF")); self.pdf.setStrokeColor(HexColor(color)); self.pdf.setLineWidth(2.3)
        self.pdf.circle(x,self.h-y,r,stroke=1,fill=1)
        self.svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="white" stroke="{color}" stroke-width="2.3"/>')
        self.text(x,y+9,label,30,color,True)

    def save(self):
        self.pdf.save()
        (OUT/(self.name+".svg")).write_text("\n".join(self.svg+["</svg>"]),encoding="utf-8")
        import pymupdf
        with pymupdf.open(OUT/(self.name+".pdf")) as doc:
            doc[0].get_pixmap(matrix=pymupdf.Matrix(4200/self.w,4200/self.w),alpha=False).save(OUT/(self.name+"_4k.png"))


def overview(audit):
    f=Figure("sepreformer_ltrr_architecture",3600,2160,
             "SepReformer + LTRR | detailed architecture",
             "Current shared 16 kHz configuration · shape probe B=2, J=2, N=64,000 · stage indices follow Python ModuleList order")
    f.panel(35,180,390,1420,"Waveform front end","[B,C,T] unless stated",BLUE)
    f.panel(520,180,540,1420,"Separation encoder","4 downsampling stages + bottleneck",ORANGE)
    f.panel(1130,180,385,1420,"Speaker split","One shared module; five calls",PURPLE)
    f.panel(1590,180,590,1420,"Reconstruction decoder","Execute D0 → D1 → D2 → D3 (upward)",GREEN)
    f.panel(2280,180,710,1420,"LTRR · final output only","One shared block for all speaker branches",PURPLE)
    f.panel(3070,180,500,1420,"Waveform reconstruction","Shared speaker-wise main head",BLUE)
    f.text(1322,298,"Conv1d 128 → 1024, k=1\nGLU channels: 1024 → 512\nConv1d 512 → 256, k=1\nview [B,256,T] → [BJ,128,T]\nGroupNorm(1,128)",19,PURPLE)
    f.text(230,326,"Mixture x: [B,N] = [2,64000]",22,bold=True)
    f.path([(230,345),(230,380)])
    f.box(65,380,330,100,"Waveform right-padding","N → Npad; here 64000",BLUE,22)
    f.path([(230,480),(230,550)])
    f.box(65,550,330,130,"AudioEncoder","Conv1d 1 → 256\nk=32, s=8, p=0; GELU",BLUE,22)
    f.text(230,720,"E: [2,256,7997]",23)
    f.path([(230,680),(230,695)]); f.path([(230,733),(230,780)])
    f.box(65,780,330,135,"FeatureProjector","GroupNorm(1,256)\nConv1d 256 → 128, k=1",BLUE,22)
    f.text(230,954,"[2,128,7997]",23)
    f.path([(230,915),(230,928)]); f.path([(230,968),(230,1010)])
    f.box(65,1010,330,100,"Separator.pad_signal","Right-pad T0 → Tp",BLUE,22)
    f.text(230,1150,"[2,128,8000]",23)
    f.path([(230,1110),(230,1125)]);f.path([(230,1165),(230,1220),(465,1220),(465,305),(790,305),(790,350)])
    f.text(65,1300,"T0 = (Npad − 32)/8 + 1\nTp = 16 × ceil(T0/16)\nHere: 7997 → 8000",22,anchor="start")
    f.text(65,1440,"Training crop: 4 seconds\nF0=256; F=128; J=2\nBJ = 4 for this probe",22,anchor="start")

    # Encoder descends; decoder ascends, keeping scale-matched skips horizontal.
    lengths=[8000,4000,2000,1000]
    for i,t in enumerate(lengths):
        y=350+260*i
        f.box(550,y,480,90,f"E{i} · (Global → Local) ×2",f"enc_stages[{i}]; [2,128,{t}]",ORANGE,21)
        f.path([(790,y+90),(790,y+125)],arrow=False);f.dot(790,y+105)
        f.path([(790,y+105),(1100,y+105),(1100,y+140),(1150,y+140)],PURPLE,True)
        f.box(580,y+125,420,75,"DownConv + BN + GELU","DWConv k=5, s=2, p=2",ORANGE,21)
        if i<3: f.path([(790,y+200),(790,y+260)])
        else: f.path([(790,y+200),(790,1440)])
        f.box(1150,y+102,345,76,f"Split S{i}",f"[2,128,{t}] → [4,128,{t}]",PURPLE,18)
        f.path([(1495,y+140),(1630,y+140)],PURPLE,True)
        # Stage D3/D2/D1/D0 at matching output resolution.
        d=3-i
        f.box(1630,y,510,90,f"D{d} · (Global → Local → CS) ×3",f"dec_stages[{d}]; [4,128,{t}]",GREEN,21)
        f.box(1630,y+117,510,63,"Concat channels → Conv1d 256 → 128","k=1; simple_fusion["+str(d)+"]",GREEN,20)
        # The skip enters the fusion input, not the stage output.
        f.box(1690,y+203,390,48,"Nearest upsample ×2",color=GREEN,size=22)
        f.path([(1885,y+203),(1885,y+180)],GREEN)
        f.path([(1885,y+117),(1885,y+90)],GREEN)
        if i<3: f.path([(1885,y+260),(1885,y+251)],GREEN)
    # Bottleneck has the same two Global/Local pairs, without downsampling.
    f.box(550,1440,480,100,"bottleneck_G: (Global → Local) ×2","No DownConv; [2,128,500]",ORANGE,22)
    f.path([(1030,1490),(1150,1490)])
    f.box(1150,1440,345,100,"Split bottleneck","[2,128,500] → [4,128,500]",PURPLE,18)
    f.path([(1495,1490),(1885,1490),(1885,1381)],GREEN)
    f.text(1885,1545,"Decoder input X0: [4,128,500]",23,GREEN)
    f.path([(1885,350),(1885,300),(2610,300),(2610,335)],GREEN)
    f.box(2380,335,460,62,"Y = [BJ,128,Tp] = [4,128,8000]",color=PURPLE,size=22)
    # LTRR outer bypass uses Y before the bottleneck projection.
    f.path([(2610,397),(2610,448)],PURPLE)
    f.dot(2610,419);f.path([(2610,419),(2935,419),(2935,1480),(2634,1480)],PURPLE)
    f.text(2914,925,"Y",25,PURPLE)
    f.box(2390,448,440,118,"Bottleneck projection","Channel LN → Conv1d 128 → 64\nk=1 → PReLU",PURPLE,22)
    f.path([(2610,566),(2610,600)],PURPLE,arrow=False);f.dot(2610,600)
    f.text(2610,631,"z: [4,64,8000]",23)
    f.path([(2610,600),(2330,600),(2330,1165),(2586,1165)],PURPLE)
    f.path([(2610,646),(2470,646),(2470,685)],PURPLE)
    f.path([(2610,646),(2750,646),(2750,685)],PURPLE)
    f.path([(2610,631+7),(2610,646)],PURPLE,arrow=False)
    f.box(2370,685,200,86,"Conv-U (u)","64 → 64",PURPLE,22)
    f.box(2650,685,200,86,"Conv-U (v)","64 → 64",PURPLE,22)
    f.text(2610,809,"Independent branch weights",20,GRAY)
    f.path([(2750,771),(2750,845)],PURPLE)
    f.box(2600,845,300,104,"Dense dilated memory","d = [1,2,4,8]\nm: [4,64,8000]",PURPLE,21)
    f.path([(2470,771),(2470,1010),(2586,1010)],PURPLE)
    f.path([(2750,949),(2750,1010),(2634,1010)],PURPLE)
    f.op(2610,1010,"×");f.text(2695,1055,"u × m",23,PURPLE)
    f.path([(2610,1034),(2610,1080)],PURPLE)
    f.box(2440,1080,340,54,"Dropout p=0.05",color=PURPLE,size=23)
    f.path([(2610,1134),(2610,1141)],PURPLE);f.op(2610,1165,"+")
    f.text(2660,1194,"h = z + Dropout(u × m)",18,PURPLE,anchor="start")
    f.path([(2610,1189),(2610,1225)],PURPLE)
    f.box(2390,1225,440,101,"Output projection","Channel LN → Conv1d 64 → 128\nk=1; ΔY: [4,128,8000]",PURPLE,21)
    f.path([(2610,1326),(2610,1370)],PURPLE)
    f.box(2440,1370,340,58,"× α  (learnable; init 0.05)",color=PURPLE,size=21)
    f.path([(2610,1428),(2610,1456)],PURPLE);f.op(2610,1480,"+")
    f.text(2610,1552,"Y′ = Y + αΔY: [4,128,8000]",23,PURPLE,True)
    f.path([(2610,1504),(2610,1520)],PURPLE)
    f.path([(2610,1568),(2610,1590),(3035,1590),(3035,445),(3090,445)],PURPLE)
    f.box(3090,335,470,220,"Main OutputLayer","Crop Tp=8000 → T0=7997\nLinear 128 → 512\nGLU: 512 → 256\nLinear 256 → 256\nReshape → [J,B,256,T0]",BLUE,23)
    f.path([(3325,555),(3325,578)]);f.text(3325,605,"[2,2,256,7997]",23)
    f.path([(3325,620),(3325,660)])
    f.box(3090,660,470,138,"AudioDecoder · shared across J","ConvTranspose1d 256 → 1\nk=32, s=8, p=0\nCrop waveform to N=64000",BLUE,23)
    f.path([(3325,798),(3325,845),(3185,845),(3185,900)])
    f.path([(3325,845),(3465,845),(3465,900)]);f.dot(3325,845)
    f.box(3090,900,190,80,"s1","[2,64000]",BLUE,23)
    f.box(3370,900,190,80,"s2","[2,64000]",BLUE,23)
    f.text(3325,1040,"Python output: list of J tensors",22,BLUE)
    f.text(3325,1076,"Main head: masking=False",22,BLUE,True)
    f.path([(3100,1140),(3540,1140)],GRAY,arrow=False,width=1.5)
    f.text(3325,1200,"Per-speaker decoder input:\n[B,256,T0] = [2,256,7997]\n\nDecoded length:\n(7997 − 1) × 8 + 32 = 64000",23,BLUE)
    f.text(3325,1440,f"Whole model: {audit['parameters']['total']:,} parameters\n(including auxiliary heads)\nLTRR only: {audit['parameters']['ltrr']:,}",22,PURPLE)
    # Bottom legends and real auxiliary path, separate from main LTRR path.
    f.panel(35,1650,500,455,"Notation & implementation","Shapes are hook-verified, not estimated",BLUE)
    f.text(65,1770,"B=2 training examples; J=2 speakers\nBJ=4; F=128; F0=256; Cb=64\nT0=7997; Tp=8000; R=4\n\nSolid: main computation / residual\nDashed purple: scale-matched skip\n× in LTRR: elementwise product\nNo extra LTRR at D0, D1 or D2",22,anchor="start")
    f.panel(575,1650,2415,455,"Auxiliary heads · return_aux=True only","Four separate OutputLayer/AudioDecoder pairs; this path does not pass through LTRR",GRAY)
    f.box(610,1785,530,160,"Xr before decoder stage Dr","r=0,1,2,3\n[BJ,128,500/1000/2000/4000]",GRAY,23)
    f.path([(1140,1865),(1190,1865)],GRAY)
    f.box(1190,1785,360,160,"Nearest interpolate","T → T0=7997\n[BJ,128,7997]",GRAY,23)
    f.path([(1550,1865),(1600,1865)],GRAY)
    f.box(1600,1785,665,160,"Aux OutputLayer: Linear → GLU → Linear","128 → 512 → 256 → 256\nReLU(mask) × E (repeated; elementwise)\n[J,B,256,7997]",GRAY,23)
    f.path([(2265,1865),(2315,1865)],GRAY)
    f.box(2315,1785,630,160,"Per-head AudioDecoder → crop to N","ConvTranspose1d 256 → 1; k=32, s=8\nEach head: list of 2 × [2,64000]",GRAY,23)
    f.text(1930,1995,"E = encoder_output [2,256,7997] supplies auxiliary masking; the main head uses E only to obtain T0.",22,GRAY)
    f.text(1782,2050,"Global = EGA → GCFN; Local = CLA → GCFN; CS = speaker attention → GCFN. All attention: 8 heads, 128 features.",22,GRAY)
    f.text(1782,2083,"Shared relative-key embedding: 4000 × 16; pos_k [500,500,16]. Local CLA kernel=65. Dropout in these blocks: 0.05.",21,GRAY)
    f.save()


def refinement(audit):
    f=Figure("ltrr_refinement_detail",2700,2110,"LTRR | residual refinement and the two Conv-U branches",
             "Lightweight Temporal Residual Refinement · original active NoGate operations · only after the final reconstruction stage")
    f.panel(40,175,1260,1710,"LTRR.forward(Y)","Input and output: [BJ,128,Tp]; current crop: [4,128,8000]",PURPLE)
    f.panel(1390,175,1270,1710,"ConvU.forward(a)","Instantiated twice: conv_u and conv_v have independent parameters",GREEN)
    f.box(360,300,570,82,"Y: [BJ,128,Tp]",color=PURPLE,size=29)
    f.path([(645,382),(645,450)],PURPLE);f.dot(645,414)
    f.path([(645,414),(1240,414),(1240,1720),(673,1720)],PURPLE)
    f.text(1195,1100,"Y",30,PURPLE)
    f.box(260,450,770,145,"Bottleneck projection","ChannelLayerNorm(128), ε=1e−8\nConv1d 128 → 64, k=1 → PReLU",PURPLE,28)
    f.path([(645,595),(645,650)],PURPLE,arrow=False);f.dot(645,650)
    f.text(645,697,"z: [BJ,64,Tp] = [4,64,8000]",29)
    f.path([(645,650),(120,650),(120,1300),(617,1300)],PURPLE)
    f.path([(645,712),(645,740)],PURPLE,arrow=False)
    f.path([(645,740),(405,740),(405,780)],PURPLE)
    f.path([(645,740),(900,740),(900,780)],PURPLE)
    f.box(245,780,320,100,"Conv-U branch u","u = ConvU(z)",PURPLE,28)
    f.box(740,780,320,100,"Conv-U branch v","v = ConvU(z)",PURPLE,28)
    f.path([(900,880),(900,950)],PURPLE)
    f.box(685,950,430,140,"DenseDilatedFSMN1D","m = Memory(v)\n[BJ,64,Tp]",PURPLE,28)
    f.path([(405,880),(405,1150),(617,1150)],PURPLE)
    f.path([(900,1090),(900,1150),(673,1150)],PURPLE)
    f.op(645,1150,"×",r=28);f.text(815,1190,"Elementwise u × m",27,PURPLE)
    f.path([(645,1178),(645,1210)],PURPLE)
    f.box(470,1210,350,48,"Dropout p=0.05",color=PURPLE,size=27)
    f.path([(645,1258),(645,1272)],PURPLE)
    f.op(645,1300,"+",r=28)
    f.text(850,1320,"h = z + Dropout(u × m)",26,PURPLE)
    f.path([(645,1328),(645,1370)],PURPLE)
    f.box(260,1370,770,130,"Output projection","ChannelLayerNorm(64), ε=1e−8\nConv1d 64 → 128, k=1",PURPLE,28)
    f.text(645,1545,"ΔY: [BJ,128,Tp]",29)
    f.path([(645,1500),(645,1513)],PURPLE);f.path([(645,1560),(645,1590)],PURPLE)
    f.box(390,1590,510,65,"Scale by learnable scalar α",color=PURPLE,size=28)
    f.text(1050,1632,"α init = 0.05",27,PURPLE)
    f.path([(645,1655),(645,1692)],PURPLE);f.op(645,1720,"+",r=28)
    f.path([(645,1748),(645,1780)],PURPLE)
    f.box(285,1780,720,70,"Y′ = Y + αΔY : [BJ,128,Tp]",color=PURPLE,size=28)
    # Conv-U: show every active operation and the branch-local residual.
    f.box(1700,300,650,82,"a: [BJ,64,Tp]",color=GREEN,size=30)
    f.path([(2025,382),(2025,455)],GREEN);f.dot(2025,415)
    f.path([(2025,415),(1470,415),(1470,1450),(1997,1450)],GREEN)
    ops=[(455,110,"ChannelLayerNorm(64)","Normalize channels at each time; ε=1e−8"),
         (640,95,"Pointwise Conv1d","64 → 64; k=1, s=1, p=0; bias=True"),
         (810,80,"SiLU",""),
         (970,145,"Depthwise Conv1d","64 → 64; groups=64; k=3\ns=1, p=1, dilation=1; bias=True"),
         (1220,90,"Dropout","p=0.05")]
    for index,(y,h,title,detail) in enumerate(ops):
        f.box(1590,y,870,h,title,detail,GREEN,28)
        if index<len(ops)-1: f.path([(2025,y+h),(2025,ops[index+1][0])],GREEN)
    f.path([(2025,1310),(2025,1422)],GREEN);f.op(2025,1450,"+",GREEN,28)
    f.path([(2025,1478),(2025,1570)],GREEN)
    f.box(1590,1570,870,100,"ConvU(a) = a + Dropout(DWConv(SiLU(PWConv(LN(a)))))",color=GREEN,size=23)
    f.text(2025,1740,"Shape preserved at every operation: [BJ,64,Tp]",29,GREEN)
    f.text(2025,1795,"Both u and v contain this residual connection.",27,GREEN)
    f.text(55,1950,"The outer bypass is Y; the inner bypass is z; each Conv-U has its own input bypass. Memory itself has no extra residual addition.",26,anchor="start")
    f.text(55,1998,"PReLU uses one learnable parameter (PyTorch default). LTRR convolutions have bias=True. No speaker-guided gate is instantiated.",26,anchor="start")
    f.text(55,2046,f"Verified parameters: LTRR = {audit['parameters']['ltrr']:,}; whole model including auxiliary heads = {audit['parameters']['total']:,}.",26,PURPLE,True,anchor="start")
    f.save()


def memory(audit):
    f=Figure("ltrr_dense_memory_detail",3400,2370,"LTRR | DenseDilatedFSMN1D, unrolled exactly as implemented",
             "Four sequential memory states · concatenation is along channels · every state has shape [BJ,64,Tp] = [4,64,8000]")
    f.box(1170,180,1060,140,"Input v → ffn_in → x0","ChannelLayerNorm(64), ε=1e−8 → Conv1d 64 → 64, k=1 → PReLU\nx0: [BJ,64,Tp]",PURPLE,26)
    centers=[410,1240,2070,2900]
    colors=[BLUE,ORANGE,GREEN,PURPLE]
    for i,cx in enumerate(centers):
        f.rect(cx-345,800,690,760,colors[i],PALE[colors[i]],dashed=True)
    # x0 bus feeds each concat. Each y bus feeds only subsequent concatenations.
    f.path([(1700,320),(1700,385),(500,385),(500,470)],BLUE,arrow=False)
    f.path([(500,470),(2990,470)],BLUE,arrow=False)
    f.text(650,450,"x0 (64 channels) → all four concatenations",24,BLUE,anchor="start")
    for cx in centers:
        f.dot(cx+90,470,BLUE);f.path([(cx+90,470),(cx+90,830)],BLUE)
    # States originate after dropout, travel up the right of their own stage,
    # then enter only later stages. Distinct ports preserve direction clearly.
    for i in range(3):
        cx=centers[i]; bus=530+i*70; color=colors[i]
        lane=cx+385
        startx=centers[i+1]+30-i*60
        endx=centers[-1]+30-i*60
        f.path([(cx,1500),(lane,1500),(lane,bus),(endx,bus)],color,dashed=True,arrow=False)
        for j in range(i+1,4):
            target=centers[j]+30-i*60
            f.dot(target,bus,color);f.path([(target,bus),(target,830)],color,dashed=True)
        f.text(startx+35,bus-15,f"y{i+1} → later stages",23,color,anchor="start")
    for i,(cx,color,d) in enumerate(zip(centers,colors,[1,2,4,8])):
        states=["x0"]+[f"y{j}" for j in range(1,i+1)]
        cin=64*(i+1)
        f.box(cx-300,830,600,100,f"State y{i+1}: Concat["+", ".join(states)+"]",f"[BJ,{cin},Tp]",color,27)
        f.path([(cx,930),(cx,975)],color)
        f.box(cx-300,975,600,100,f"Pointwise Conv1d {cin} → 64","k=1, s=1, p=0; bias=True",color,27)
        f.path([(cx,1075),(cx,1120)],color)
        f.box(cx-300,1120,600,120,"Depthwise Conv1d 64 → 64",f"k=3; dilation={d}; padding={d}\ngroups=64; stride=1; bias=True",color,27)
        f.path([(cx,1240),(cx,1280)],color)
        f.box(cx-230,1280,460,58,"PReLU",color=color,size=29)
        f.path([(cx,1338),(cx,1380)],color)
        f.box(cx-230,1380,460,58,"Dropout p=0.05",color=color,size=29)
        f.path([(cx,1438),(cx,1500)],color,arrow=False);f.dot(cx,1500,color)
        f.text(cx-18,1537,f"y{i+1}",28,color,anchor="end")
        f.path([(cx,1500),(cx,1660)],color,arrow=False)
    f.path([(410,1660),(2900,1660)],PURPLE,arrow=False)
    for cx in centers: f.dot(cx,1660,PURPLE)
    f.dot(1700,1660,PURPLE);f.path([(1700,1660),(1700,1760)],PURPLE)
    f.box(1170,1760,1060,120,"Concat[y1, y2, y3, y4]","[BJ,256,Tp] · x0 is NOT included in this final concatenation",PURPLE,28)
    f.path([(1700,1880),(1700,1940)],PURPLE)
    f.box(1170,1940,1060,110,"out_proj: Conv1d 256 → 64","k=1, s=1, p=0; bias=True",PURPLE,28)
    f.path([(1700,2050),(1700,2110)],PURPLE)
    f.box(1290,2110,820,70,"m: [BJ,64,Tp] = [4,64,8000]",color=PURPLE,size=30)
    f.text(1700,2255,"Solid top bus: x0. Dashed feedback buses: earlier states to later stages. Dots denote junctions; crossings without dots do not connect.",24)
    f.text(1700,2305,"All arrows follow the acyclic forward computation. The returned memory m is multiplied by u in TemporalInteraction; it is not added to v here.",24)
    f.save()


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--extra-pythonpath",type=Path)
    args=parser.parse_args()
    if args.extra_pythonpath: sys.path.insert(0,str(args.extra_pythonpath))
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    choices=[(Path("C:/Windows/Fonts/arial.ttf"),Path("C:/Windows/Fonts/arialbd.ttf")),
             (Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"))]
    normal,bold=next(pair for pair in choices if all(p.exists() for p in pair))
    pdfmetrics.registerFont(TTFont("Diagram",str(normal)))
    pdfmetrics.registerFont(TTFont("DiagramBold",str(bold)))
    audit=json.loads((OUT/"ltrr_shape_audit.json").read_text())
    for name,digest in audit["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, f"Source changed: {name}; rerun audit and review diagram."
    assert audit["parameters"] == {"total":14805191,"ltrr":89031}
    overview(audit);refinement(audit);memory(audit)
    import pymupdf
    combined=pymupdf.open()
    for name in ["sepreformer_ltrr_architecture","ltrr_refinement_detail","ltrr_dense_memory_detail"]:
        with pymupdf.open(OUT/(name+".pdf")) as source: combined.insert_pdf(source)
    combined.save(OUT/"ltrr_architecture_complete.pdf")
    print("Written: 3 SVGs, 3 PDFs, 3 PNG previews, and ltrr_architecture_complete.pdf")


if __name__ == "__main__":
    main()
