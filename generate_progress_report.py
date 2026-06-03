# -*- coding: utf-8 -*-
"""
BCI-VR 프로젝트 진척 현황 PDF 보고서 생성기
macOS 시스템 한글 폰트를 사용하여 PDF를 생성합니다.

실행 방법:
    cd /Volumes/a3122a1/MI-BCI
    .venv/bin/python generate_progress_report.py
"""

import os, sys, io
from pathlib import Path

# ── 의존성 확인 ───────────────────────────────────────────────────────
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Pillow 설치 중...")
    os.system(f"{sys.executable} -m pip install Pillow reportlab -q")
    from PIL import Image, ImageDraw, ImageFont

try:
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage
    from reportlab.lib.pagesizes import A4 as RL_A4
except ImportError:
    print("reportlab 설치 중...")
    os.system(f"{sys.executable} -m pip install reportlab -q")
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage
    from reportlab.lib.pagesizes import A4 as RL_A4

# ── 한글 폰트 자동 탐색 ───────────────────────────────────────────────
KOREAN_FONT_CANDIDATES = [
    "/System/Library/Fonts/AppleSDGothicNeo.ttc",
    "/Library/Fonts/NanumGothic.ttf",
    "/Library/Fonts/NanumGothicBold.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/AppleGothic.ttf",
    os.path.expanduser("~/Library/Fonts/NanumGothic.ttf"),
    "/Library/Fonts/Arial Unicode.ttf",
]

FONT_PATH = None
for p in KOREAN_FONT_CANDIDATES:
    if os.path.exists(p):
        FONT_PATH = p
        print(f"✅ 한글 폰트 발견: {p}")
        break

if FONT_PATH is None:
    print("❌ 한글 폰트를 찾지 못했습니다.")
    print("   /Library/Fonts/ 에 NanumGothic.ttf 를 복사한 후 다시 실행하세요.")
    sys.exit(1)

OUTPUT = str(Path(__file__).parent / "BCI_VR_Progress_Report_20260525.pdf")

# ── 페이지 설정 ───────────────────────────────────────────────────────
DPI    = 180
A4_W   = int(210 * DPI / 25.4)
A4_H   = int(297 * DPI / 25.4)
MARGIN = int(20 * DPI / 25.4)
CW     = A4_W - 2 * MARGIN   # content width

# ── 색상 ─────────────────────────────────────────────────────────────
NAVY   = (27,  58, 107)
TEAL   = (13, 124, 124)
LIGHT  = (238, 243, 251)
GREEN  = (46,  125,  50)
AMBER  = (230,  81,   0)
RED    = (183,  28,  28)
GREY   = (84,  110, 122)
WHITE  = (255, 255, 255)
BLACK  = (20,  20,  20)
LT_GRN = (232, 245, 233)
LT_AMB = (255, 248, 225)

# ── 폰트 팩토리 ──────────────────────────────────────────────────────
def f(pt, bold=False):
    """pt 크기의 PIL 폰트 반환"""
    size = int(pt * DPI / 72)
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()

F_TITLE = f(20)
F_H1    = f(14)
F_H2    = f(11)
F_BODY  = f(10)
F_SMALL = f(9)
F_TINY  = f(8)

# ── 드로잉 유틸 ──────────────────────────────────────────────────────
def tw(draw, text, fnt):
    b = draw.textbbox((0,0), text, font=fnt)
    return max(b[2] - b[0], 1)

def th(draw, text, fnt):
    b = draw.textbbox((0,0), text, font=fnt)
    return max(b[3] - b[1], 1)

def rect(draw, x, y, w, h, fill):
    draw.rectangle([x, y, x+w, y+h], fill=fill)

def hline(draw, y, color=GREY, width=1):
    draw.line([(MARGIN, y), (A4_W-MARGIN, y)], fill=color, width=width)

def new_page():
    img  = Image.new("RGB", (A4_W, A4_H), WHITE)
    draw = ImageDraw.Draw(img)
    return img, draw

def header(draw, page_n):
    h = int(16 * DPI / 25.4)
    rect(draw, 0, 0, A4_W, h, NAVY)
    draw.text((MARGIN, int(3*DPI/25.4)), "BCI-VR Project  |  Progress Report",
              font=F_BODY, fill=WHITE)
    s = "2026-05-25"
    draw.text((A4_W - MARGIN - tw(draw,s,F_BODY), int(3*DPI/25.4)),
              s, font=F_BODY, fill=WHITE)
    return h

def footer(draw, page_n):
    fy = A4_H - int(13*DPI/25.4)
    hline(draw, fy, (176,190,197))
    s = f"Page {page_n}  |  Kyung Hee University — Motor Imagery BCI-VR Research Team"
    draw.text(((A4_W - tw(draw,s,F_TINY))//2, fy+3), s, font=F_TINY, fill=GREY)

def sec_title(draw, y, text):
    draw.text((MARGIN, y), text, font=F_H1, fill=NAVY)
    y += th(draw, text, F_H1) + 4
    hline(draw, y, NAVY, 2)
    return y + 8

def sub_title(draw, y, text):
    draw.text((MARGIN, y), text, font=F_H2, fill=TEAL)
    return y + th(draw, text, F_H2) + 5

def body(draw, y, text, color=BLACK, indent=0):
    draw.text((MARGIN+indent, y), text, font=F_BODY, fill=color)
    return y + th(draw, text, F_BODY) + 4

# ── 텍스트 자동 줄바꿈 드로잉 ────────────────────────────────────────
def draw_wrapped(draw, x, y, text, fnt, max_w, color=BLACK):
    """max_w 안에서 줄바꿈하여 텍스트 드로잉, 최종 y 반환"""
    line = ""
    for ch in text:
        if tw(draw, line+ch, fnt) <= max_w:
            line += ch
        else:
            if line:
                draw.text((x, y), line, font=fnt, fill=color)
                y += th(draw, line, fnt) + 2
            line = ch
    if line:
        draw.text((x, y), line, font=fnt, fill=color)
        y += th(draw, line, fnt) + 2
    return y

# ── 진척도 바 ─────────────────────────────────────────────────────────
def draw_bar(draw, x, y, pct, w):
    fill = GREEN if pct>=100 else (TEAL if pct>=60 else AMBER)
    bh = int(8*DPI/72)
    rect(draw, x, y, w, bh, (222,227,232))
    rect(draw, x, y, max(2, int(w*pct/100)), bh, fill)
    label = f"{pct}%"
    lc = GREEN if pct>=100 else (TEAL if pct>=60 else AMBER)
    draw.text((x, y+bh+2), label, font=F_TINY, fill=lc)
    return y + bh + th(draw, label, F_TINY) + 4

# ── 테이블 그리기 ─────────────────────────────────────────────────────
def draw_table(draw, x, y, headers, rows, col_widths,
               hdr_bg=NAVY, row_bgs=None, pad=6):
    rb  = row_bgs or [LIGHT, WHITE]
    total_w = sum(col_widths)
    LINE = (176, 190, 197)

    def calc_row_h(cells, fnt):
        rh = 0
        for i, cell in enumerate(cells):
            if isinstance(cell, str):
                cw = col_widths[i] - 2*pad
                lines, cur = [], ""
                for ch in cell:
                    if tw(draw, cur+ch, fnt) > cw:
                        if cur: lines.append(cur)
                        cur = ch
                    else:
                        cur += ch
                if cur: lines.append(cur)
                h = max(1,len(lines)) * (th(draw,"가",fnt)+2)
                rh = max(rh, h)
        return max(int(22*DPI/72), rh + 2*pad)

    def draw_row(cells, ry, bg, fnt, tc=BLACK, is_bar_row=False):
        rh = calc_row_h(cells, fnt)
        rect(draw, x, ry, total_w, rh, bg)
        cx = x
        for i, cell in enumerate(cells):
            cw = col_widths[i]
            if isinstance(cell, str):
                max_w = cw - 2*pad
                ty = ry + pad
                cur = ""
                for ch in cell:
                    if tw(draw, cur+ch, fnt) > max_w:
                        if cur:
                            draw.text((cx+pad, ty), cur, font=fnt, fill=tc)
                            ty += th(draw, cur, fnt) + 2
                        cur = ch
                    else:
                        cur += ch
                if cur:
                    draw.text((cx+pad, ty), cur, font=fnt, fill=tc)
            elif isinstance(cell, tuple) and cell[0] == "bar":
                pct  = cell[1]
                bw   = cw - 2*pad
                bh   = int(8*DPI/72)
                fill = GREEN if pct>=100 else (TEAL if pct>=60 else AMBER)
                by   = ry + rh//2 - bh - 4
                rect(draw, cx+pad, by, bw, bh, (222,227,232))
                rect(draw, cx+pad, by, max(2,int(bw*pct/100)), bh, fill)
                lbl = f"{pct}%"
                lc  = GREEN if pct>=100 else (TEAL if pct>=60 else AMBER)
                draw.text((cx+pad, by+bh+2), lbl, font=F_TINY, fill=lc)
            cx += cw
        draw.rectangle([x,ry,x+total_w,ry+rh], outline=LINE, width=1)
        cx = x
        for cw in col_widths[:-1]:
            cx += cw
            draw.line([(cx,ry),(cx,ry+rh)], fill=LINE, width=1)
        return ry + rh

    y = draw_row(headers, y, hdr_bg, F_SMALL, WHITE)
    for ri, row in enumerate(rows):
        y = draw_row(row, y, rb[ri%len(rb)], F_SMALL, BLACK)
    return y

# ═══════════════════════════════════════════════════════════════════════
# 페이지 1 — 개요 + 파이프라인
# ═══════════════════════════════════════════════════════════════════════
img1, d1 = new_page()
hh = header(d1, 1)
footer(d1, 1)
y = hh + int(8*DPI/25.4)

# 타이틀 블록
tb_h = int(46*DPI/25.4)
rect(d1, MARGIN, y, CW, tb_h, NAVY)
t1 = "EEG+sEMG Hybrid BCI-VR System"
t2 = "Project Progress Report  —  2026년 5월 25일"
t3 = "Kyung Hee University  |  Motor Imagery Research Team  |  JNE Submission Target"
cx = A4_W // 2
d1.text((cx - tw(d1,t1,F_TITLE)//2, y+5),              t1, font=F_TITLE, fill=WHITE)
d1.text((cx - tw(d1,t2,F_H2)//2,    y+int(26*DPI/72)), t2, font=F_H2,    fill=(200,220,255))
d1.text((cx - tw(d1,t3,F_TINY)//2,  y+int(37*DPI/72)), t3, font=F_TINY,  fill=(180,210,240))
y += tb_h + int(7*DPI/25.4)

# 1. 프로젝트 개요
y = sec_title(d1, y, "1. 프로젝트 개요")
c1, c2 = int(0.27*CW), int(0.73*CW)
y = draw_table(d1, MARGIN, y,
    ["항목", "내용"],
    [["데이터셋","GigaDB Cho et al. 2017 (DOI: 10.5524/100295)"],
     ["피험자",  "52명 (여 19명, 평균 연령 24.8세)"],
     ["신호",    "EEG 64ch + sEMG 4ch @ 512 Hz"],
     ["분류",    "Left MI vs. Right MI (2-class)"],
     ["응용",    "Meta Quest 3S VR 아바타 양팔 실시간 제어"],
     ["투고",    "Journal of Neural Engineering (JNE, IF ~5.0)"]],
    [c1,c2]) + int(5*DPI/25.4)

y = sub_title(d1, y, "팀 구성")
y = draw_table(d1, MARGIN, y,
    ["팀원","담당 스테이지","세부 역할"],
    [["이원석 (Member A)","S1/S2/S3/S5","Data Loading, Preprocessing, Model Training, Backend + VR Integration (담당자)"],
     ["차준엽","S4","XAI — DeepSHAP Analysis"],
     ["정민규","S4","XAI — Grad-CAM Topomap"],
     ["남동연","S4/S6","XAI — ERD Validation, Statistical Eval"]],
    [int(0.24*CW),int(0.18*CW),int(0.58*CW)],
    hdr_bg=TEAL,
    row_bgs=[(224,242,241),WHITE,LIGHT,WHITE]) + int(7*DPI/25.4)

# 2. 파이프라인
y = sec_title(d1, y, "2. 전체 파이프라인 진척 현황")
stages = [
    ["S1","Data Loading",    "52개 .mat → HDF5 변환",               100,"완료"],
    ["S2","Preprocessing",   "EEG ICA + sEMG RMS 전처리 (52명)",    100,"완료"],
    ["S3","Model Training",  "LOSO 52-fold, A100 (v4 baseline)",   100,"완료"],
    ["S4","XAI Analysis",    "DeepSHAP / Grad-CAM / ERD (타 팀원)", 20,"진행 중"],
    ["S5","Real-time BCI-VR","WebSocket + Unity + Meta Quest 3S",   65,"진행 중"],
    ["S6","Statistical Eval","Wilcoxon + ITR (타 팀원)",             30,"진행 중"],
    ["—", "Paper Writing",   "JNE 투고용 논문 작성",                  40,"진행 중"],
]
sc = GREEN if True else AMBER
y = draw_table(d1, MARGIN, y,
    ["Stage","명칭","내용","진척도","상태"],
    [[s[0],s[1],s[2],("bar",s[3]),s[4]] for s in stages],
    [int(0.07*CW),int(0.19*CW),int(0.40*CW),int(0.21*CW),int(0.13*CW)])

# ═══════════════════════════════════════════════════════════════════════
# 페이지 2 — 모델 성능 + S5 상세
# ═══════════════════════════════════════════════════════════════════════
img2, d2 = new_page()
header(d2, 2); footer(d2, 2)
y = int(22*DPI/25.4)

y = sec_title(d2, y, "3. 모델 성능 (v4 Baseline, LOSO-52)")
y = sub_title(d2, y, "HybridBCIModel 구조")
y = body(d2, y, "EEGNetEncoder (64ch EEG → 256-dim)  +  EMGBiLSTMEncoder (4ch sEMG → 256-dim)")
y = body(d2, y, "+  SoftmaxAttentionFusion (256-dim)  +  Classifier (FC 256→128→2)")
y += int(2*DPI/25.4)

y = sub_title(d2, y, "성능 지표")
y = draw_table(d2, MARGIN, y,
    ["지표","값","비고"],
    [["Accuracy",    "74.2% ± 11.1%","LOSO 52-fold 평균"],
     ["F1-Macro",    "0.738",         "Left/Right 균형 지표"],
     ["Cohen's k",   "0.484",         "Fair Agreement"],
     ["ITR",         "3.44 bits/min", "30명(58%)은 5+ bits/min 달성"],
     ["추론 Latency", "~12 ms",        "ONNX CPU (FastAPI)"],
     ["전체 Latency", "~26-32 ms",     "목표 100ms 대비 충분한 여유"]],
    [int(0.25*CW),int(0.28*CW),int(0.47*CW)],
    hdr_bg=TEAL) + int(4*DPI/25.4)

y = sub_title(d2, y, "피험자별 성능 분포")
y = draw_table(d2, MARGIN, y,
    ["그룹","범위","인원","대표 피험자"],
    [["High",  "> 85%", "8명",  "s03, s04, s43 (최고 97%)"],
     ["Medium","65-85%","32명", "s22, s48 (중앙값 73%)"],
     ["Low",   "< 65%", "12명", "s07, s34 (최저 52%)"]],
    [int(0.13*CW),int(0.15*CW),int(0.11*CW),int(0.61*CW)],
    row_bgs=[LT_GRN,LIGHT,LT_AMB]) + int(7*DPI/25.4)

y = sec_title(d2, y, "4. S5 실시간 BCI-VR 상세 현황 (65%)")
y = sub_title(d2, y, "완료 항목")
y = draw_table(d2, MARGIN, y,
    ["파일 / 경로","내용","규모"],
    [["src/inference.py",             "SignalSimulator + BCIInferenceEngine + OSC VRBridge + CLI","504줄"],
     ["src/export_onnx.py",           "52명 ONNX 변환 (opset 17, 검증 포함)",                   "379줄"],
     ["src/websocket_server.py",      "asyncio WebSocket 서버 (broadcast, conf. threshold)",    "390줄"],
     ["src/test_ws_client.py",        "Python 테스트 클라이언트 (latency 리포트)",               "262줄"],
     ["unity/AvatarController.cs",    "Unity IK 팔 제어 (DirectionClass 8개, NativeWebSocket)", "413줄"],
     ["unity/Packages/manifest.json", "NativeWebSocket + Meta XR + URP 패키지 의존성",          "—"],
     ["BCI_Research/results/onnx/",   "52개 ONNX 모델 (avg latency 12.44ms)",                  "52 files"],
     ["BCI_Research/checkpoints_A/",  "best_s01.pt ~ best_s52.pt",                             "52 files"]],
    [int(0.31*CW),int(0.54*CW),int(0.15*CW)],
    row_bgs=[LT_GRN,WHITE]) + int(4*DPI/25.4)

y = sub_title(d2, y, "남은 항목  (Unity Editor / 실기기 필요)")
y = draw_table(d2, MARGIN, y,
    ["#","작업 내용","우선순위"],
    [["1","Packages/manifest.json 교체 → NativeWebSocket 자동 설치","High"],
     ["2","AvatarController.cs → Avatar GameObject 연결 (IK Target 지정)","High"],
     ["3","로컬 연결 테스트: websocket_server.py --sid 3 + Unity Play Mode","High"],
     ["4","Meta Quest 3S 빌드: Android + XR Plugin Management + URP","Medium"],
     ["5","실기기 End-to-End latency 측정 (목표 <100ms, 예상 ~25-35ms)","Medium"],
     ["6","데모 영상 촬영","Low"]],
    [int(0.06*CW),int(0.79*CW),int(0.15*CW)],
    row_bgs=[LT_AMB,WHITE])

# ═══════════════════════════════════════════════════════════════════════
# 페이지 3 — Latency + Ablation + 향후 계획
# ═══════════════════════════════════════════════════════════════════════
img3, d3 = new_page()
header(d3, 3); footer(d3, 3)
y = int(22*DPI/25.4)

y = sub_title(d3, y, "Latency 예산 (S5)")
y = draw_table(d3, MARGIN, y,
    ["구성 요소","예상 시간"],
    [["모델 추론 (ONNX CPU)","~12 ms"],
     ["WebSocket 전송 (LAN)","~2-5 ms"],
     ["Unity JSON 파싱",     "< 1 ms"],
     ["IK 계산 + URP 렌더링","~11-14 ms"],
     ["합계 (목표: < 100ms)","~26-32 ms  ←  목표 달성 가능"]],
    [int(0.47*CW),int(0.53*CW)],
    hdr_bg=GREY,
    row_bgs=[LIGHT,WHITE,LIGHT,WHITE,LT_GRN]) + int(7*DPI/25.4)

y = sec_title(d3, y, "5. 알려진 이슈 및 Ablation Study 결과")
y = sub_title(d3, y, "Right MI Bias 문제")
y = body(d3, y, "9/52명(17%)에서 Right MI 과분류 발생  (Left/Right Recall 차이 30% 이상)")
y = body(d3, y, "Bias 피험자: s01, s05, s07, s11, s12, s15, s24, s34, s36")
y = body(d3, y, "Root Cause: 52명 중 37명(71%)에서 Left MI 신호가 평균 3% 강함 → 모델 과학습")
y += int(3*DPI/25.4)

y = draw_table(d3, MARGIN, y,
    ["전략","k 손실","Bias Fix율","판정"],
    [["v4 baseline (채택)",      "0%",     "0%",   "Best performance (채택)"],
     ["v5 label smooth e=0.10", "-12.8%", "100%", "과도한 성능 하락"],
     ["v5 label smooth e=0.05", "-12.7%", "89%",  "포화 상태"],
     ["v6 class-wt w=1.2/0.8",  "-11.1%", "89%",  "손실 과다"],
     ["v6 class-wt w=1.3/0.7",  "-7.5%",  "78%",  "최적 trade-off (기각)"],
     ["v6 class-wt w=1.4/0.6",  "-8.3%",  "100%", "성능 하락"]],
    [int(0.37*CW),int(0.14*CW),int(0.15*CW),int(0.34*CW)],
    row_bgs=[LT_GRN,LIGHT,WHITE,LIGHT,WHITE,LIGHT]) + int(3*DPI/25.4)

d3.text((MARGIN, y),
    "결론: 모든 bias 보정 전략이 k 손실 7.5% 이상 유발 → v4 baseline 유지, 논문 Limitations에 기재 예정",
    font=F_SMALL, fill=RED)
y += th(d3,"가",F_SMALL) + int(5*DPI/25.4)

y = sub_title(d3, y, "미시도 해결책 (추후 시도 가능)")
for txt in [
    "1. Hemispheric Flip Augmentation: 64채널 좌우 대칭 쌍(15쌍)으로 Left MI → synthetic Right MI 생성",
    "2. Post-hoc Logit Calibration: 기존 체크포인트 재사용, logit bias 추정 후 보정 (재학습 불필요)",
    "3. Questionnaire 연계 분석: Questionnaire_results_of_52_subjects.xlsx — bias 피험자 특성 분석",
]:
    y = body(d3, y, txt)
y += int(6*DPI/25.4)

y = sec_title(d3, y, "6. 향후 계획 (Next Steps)")
y = draw_table(d3, MARGIN, y,
    ["Phase","내용","담당","우선순위"],
    [["Phase 3","S5 WebSocket + Unity 연동 완료 (남은 6개 항목)","이원석","높음"],
     ["Phase 4","S5 Meta Quest 3S 빌드 + 데모 영상 촬영",        "이원석","중간"],
     ["Phase 1","S4 XAI 파일럿 — 대표 5명 (DeepSHAP/Grad-CAM/ERD)","타 팀원","높음"],
     ["Phase 2","S4 XAI 전체 52명 확장",                         "타 팀원","높음"],
     ["Phase 5","S6 통계 + 논문 작성 (JNE 투고)",                 "전체",  "낮음"]],
    [int(0.13*CW),int(0.53*CW),int(0.17*CW),int(0.17*CW)]) + int(5*DPI/25.4)

y = sub_title(d3, y, "Known Limitations (논문 기재 예정)")
for txt in [
    "1. Right MI 과분류: 17% 피험자 (9/52명) — 학습 알고리즘 한계, bias 보정 실패",
    "2. 저성능 그룹 (52-65%, ~12명): 원인 미분석 → S6 통계 분석에서 다룰 예정",
    "3. HDF5 Replay 기반: 실시간 전극 입력(online EEG acquisition) 미구현",
    "4. 로컬 venv에 PyTorch 미설치 — Colab A100 전용 학습 환경",
]:
    y = body(d3, y, txt)

fy = A4_H - int(26*DPI/25.4)
hline(d3, fy, GREY)
note = "이 보고서는 2026-05-25 기준 프로젝트 컨텍스트를 바탕으로 자동 생성되었습니다.  이원석 — Kyung Hee University"
d3.text(((A4_W-tw(d3,note,F_TINY))//2, fy+4), note, font=F_TINY, fill=GREY)

# ── PDF 조립 (PNG → reportlab) ───────────────────────────────────────
print("PDF 생성 중...")
pages = [img1, img2, img3]

try:
    # reportlab 방식 (JPEG 코덱 불필요)
    from reportlab.platypus import SimpleDocTemplate, Image as RLImage
    from reportlab.lib.pagesizes import A4 as RL_A4

    doc = SimpleDocTemplate(OUTPUT, pagesize=RL_A4,
                            topMargin=0, bottomMargin=0,
                            leftMargin=0, rightMargin=0)
    story = []
    for pg in pages:
        buf = io.BytesIO()
        pg.save(buf, format="PNG")
        buf.seek(0)
        rl_img = RLImage(buf, width=RL_A4[0], height=RL_A4[1])
        story.append(rl_img)

    # reportlab은 마진을 frame에서 빼므로 PageTemplate으로 전체 페이지 채우기
    from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

    class FullPageDoc(BaseDocTemplate):
        def build(self, flowables, **kw):
            self._calc()
            frame = Frame(0, 0, self.pagesize[0], self.pagesize[1],
                          leftPadding=0, rightPadding=0,
                          topPadding=0, bottomPadding=0)
            self.addPageTemplates(PageTemplate(id='full', frames=[frame]))
            BaseDocTemplate.build(self, flowables, **kw)

    doc2 = FullPageDoc(OUTPUT, pagesize=RL_A4)
    story2 = []
    for pg in pages:
        buf = io.BytesIO()
        pg.save(buf, format="PNG")
        buf.seek(0)
        story2.append(RLImage(buf, width=RL_A4[0], height=RL_A4[1]))
    doc2.build(story2)
    print("  (reportlab 방식 사용)")

except Exception as e:
    print(f"  reportlab 실패: {e}, TIFF 방식 시도...")
    # TIFF 멀티페이지 → PDF 변환 (libtiff 필요)
    tmp_tiff = OUTPUT.replace(".pdf", "_tmp.tiff")
    pages[0].save(tmp_tiff, save_all=True, append_images=pages[1:],
                  compression="tiff_deflate")
    import subprocess
    subprocess.run(["tiff2pdf", "-o", OUTPUT, tmp_tiff], check=True)
    os.remove(tmp_tiff)
    print("  (tiff2pdf 방식 사용)")

size_kb = os.path.getsize(OUTPUT) / 1024
print(f"✅ 완료! {OUTPUT}")
print(f"   파일 크기: {size_kb:.0f} KB  |  페이지: {len(pages)}p")
