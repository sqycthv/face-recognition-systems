import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Access Terminal",
    page_icon="🔐",
    layout="wide"
)

DB_PATH = "database/users"
ATTEMPTS_PATH = "database/attempts"

os.makedirs(DB_PATH, exist_ok=True)
os.makedirs(ATTEMPTS_PATH, exist_ok=True)

# =========================================================
# SESSION STATE
# =========================================================
if "current_page" not in st.session_state:
    st.session_state.current_page = "Главная"

if "stats" not in st.session_state:
    st.session_state.stats = {
        "true": 0,
        "false": 0,
        "spoof": 0,
        "total": 0
    }

if "logs" not in st.session_state:
    st.session_state.logs = []

if "last_result" not in st.session_state:
    st.session_state.last_result = {
        "status": "idle",
        "title": "Ожидание сканирования",
        "detail": "Поднесите лицо к камере",
        "similarity": None,
        "variance": None,
        "blur": None,
        "confidence": None,
        "quality": None,
    }

if "settings" not in st.session_state:
    st.session_state.settings = {
        "diff_threshold": 55,
        "similarity_threshold": 45,
        "spoof_variance_threshold": 800,
        "spoof_blur_threshold": 50,
        "scan_delay": 0.008,
    }

if "is_auth" not in st.session_state:
    st.session_state.is_auth = False    

stats = st.session_state.stats
settings = st.session_state.settings

# =========================================================
# HELPERS
# =========================================================
def go(page: str):
    st.session_state.current_page = page
    st.rerun()


def login_user(username, password):
    admin_login = "admin"
    admin_password = "1234"
    return username == admin_login and password == admin_password    


def nav_button(label, page, icon):
    if st.sidebar.button(f"{icon}  {label}", use_container_width=True):
        go(page)


def bytes_to_bgr(uploaded_file):
    if uploaded_file is None:
        return None
    data = uploaded_file.getvalue()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def save_attempt_image(img, label):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{label}_{timestamp}.jpg"
    path = os.path.join(ATTEMPTS_PATH, filename)
    cv2.imwrite(path, img)
    return path


def compare_faces(img):
    best = None
    min_diff = 999999.0

    for filename in os.listdir(DB_PATH):
        path = os.path.join(DB_PATH, filename)
        db_img = cv2.imread(path)

        if db_img is None:
            continue

        try:
            img1 = cv2.resize(img, (120, 120))
            img2 = cv2.resize(db_img, (120, 120))
        except Exception:
            continue

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        diff = np.mean(cv2.absdiff(gray1, gray2))

        if diff < min_diff:
            min_diff = diff
            best = filename

    return best, min_diff


def analyze_image_quality(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = float(np.var(gray))
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(gray))
    return variance, blur, brightness


def is_spoof(img):
    variance, blur, brightness = analyze_image_quality(img)

    very_dark = brightness < 45
    very_bright = brightness > 220
    very_blurry = blur < settings["spoof_blur_threshold"]
    very_flat = variance < settings["spoof_variance_threshold"]

    suspicious = (
        (very_blurry and very_flat)
        or (very_dark and very_blurry)
        or (very_bright and very_blurry)
    )

    return suspicious, variance, blur, brightness


def draw_face_box(img):
    out = img.copy()
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    for (x, y, w, h) in faces[:1]:
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 255), 3)
        cv2.putText(
            out,
            "FACE LOCK",
            (x, max(y - 12, 24)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def add_log(result, detail, similarity):
    st.session_state.logs.insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "result": result,
            "detail": detail,
            "similarity": f"{similarity:.1f}%"
        }
    )
    st.session_state.logs = st.session_state.logs[:30]


def set_result(
    status,
    title,
    detail,
    similarity=None,
    variance=None,
    blur=None,
    confidence=None,
    quality=None
):
    st.session_state.last_result = {
        "status": status,
        "title": title,
        "detail": detail,
        "similarity": similarity,
        "variance": variance,
        "blur": blur,
        "confidence": confidence,
        "quality": quality,
    }


def get_quality_label(variance, blur):
    if variance > 2500 and blur > 180:
        return "ХОРОШЕЕ"
    if variance > 1200 and blur > 70:
        return "СРЕДНЕЕ"
    return "ПЛОХОЕ"


def get_confidence(similarity):
    if similarity >= 85:
        return "ВЫСОКАЯ"
    if similarity >= 65:
        return "СРЕДНЯЯ"
    return "НИЗКАЯ"


def get_security_level(far, frr):
    if far < 0.10 and frr < 0.20:
        return "ВЫСОКИЙ"
    if far < 0.20 and frr < 0.30:
        return "СРЕДНИЙ"
    return "НИЗКИЙ"


def reset_all():
    st.session_state.stats = {
        "true": 0,
        "false": 0,
        "spoof": 0,
        "total": 0
    }
    st.session_state.logs = []
    st.session_state.last_result = {
        "status": "idle",
        "title": "Ожидание сканирования",
        "detail": "Поднесите лицо к камере",
        "similarity": None,
        "variance": None,
        "blur": None,
        "confidence": None,
        "quality": None,
    }


# =========================================================
# STYLE
# =========================================================
st.markdown("""
<style>
:root{
    --bg1:#090b12;
    --bg2:#121524;
    --bg3:#0d1019;
    --stroke:rgba(255,255,255,0.08);
    --muted:#cbd5e1;
    --shadow:0 18px 40px rgba(0,0,0,0.34);
}

.stApp {
    background:
        radial-gradient(circle at 10% 8%, rgba(255,145,0,0.20), transparent 22%),
        radial-gradient(circle at 88% 10%, rgba(123,47,247,0.18), transparent 22%),
        radial-gradient(circle at 70% 82%, rgba(0,201,107,0.08), transparent 18%),
        linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 48%, var(--bg3) 100%);
    color: white;
    font-family: "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 0.9rem !important;
    padding-bottom: 1rem !important;
    max-width: 1460px;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, h4, h5, h6, p, label, div, span {
    color: white;
}

.topbar {
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap:20px;
    margin-bottom:16px;
    padding:14px 18px;
    border-radius:24px;
    background:linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.025));
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:var(--shadow);
    backdrop-filter: blur(12px);
}

.topbar-left {
    display:flex;
    align-items:center;
    gap:14px;
}

.topbar-dots {
    display:flex;
    gap:8px;
}

.dot {
    width:12px;
    height:12px;
    border-radius:50%;
}

.dot-red { background:#ff5f57; }
.dot-yellow { background:#ffbd2e; }
.dot-green { background:#28c840; }

.topbar-title {
    font-size:28px;
    font-weight:900;
    background:linear-gradient(90deg,#ff7a18,#ff0055,#7b2ff7);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.topbar-subtitle {
    color:var(--muted);
    font-size:14px;
}

.hero {
    text-align:center;
    margin-bottom:18px;
    padding:18px;
    border-radius:28px;
    background:
        linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
    border:1px solid rgba(255,255,255,0.08);
    box-shadow:var(--shadow);
    backdrop-filter: blur(12px);
}

.hero-title{
    font-size:42px;
    font-weight:900;
    letter-spacing:1px;
    margin-bottom:6px;
    background:linear-gradient(90deg,#ff7a18,#ff0055,#7b2ff7);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.hero-subtitle{
    color:var(--muted);
    font-size:16px;
}

.card{
    background:
        linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.03));
    border:1px solid var(--stroke);
    border-radius:24px;
    padding:22px;
    box-shadow:var(--shadow);
    backdrop-filter:blur(12px);
    margin-bottom:18px;
}

.terminal-panel{
    background:linear-gradient(180deg, rgba(17,19,31,0.98), rgba(9,11,19,0.98));
    border:1px solid rgba(255,255,255,0.08);
    border-radius:28px;
    padding:24px;
    box-shadow:var(--shadow);
    margin-bottom:18px;
}

.app-tile{
    background:
        linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.035));
    border:1px solid rgba(255,255,255,0.08);
    border-radius:24px;
    padding:22px;
    height: 235px;
    box-shadow:0 14px 28px rgba(0,0,0,0.26);
    position:relative;
    overflow:hidden;
    transition:0.25s ease;
    margin-bottom:18px;
    display:flex;
    flex-direction:column;
    justify-content:flex-start;
}

.app-tile::before {
    content:"";
    position:absolute;
    inset:0;
    border-radius:24px;
    padding:1.2px;
    background:linear-gradient(90deg, rgba(255,122,24,0.95), rgba(255,0,85,0.95), rgba(123,47,247,0.95));
    -webkit-mask:
        linear-gradient(#000 0 0) content-box,
        linear-gradient(#000 0 0);
    -webkit-mask-composite:xor;
    mask-composite: exclude;
    pointer-events:none;
}

.app-tile::after{
    content:"";
    position:absolute;
    right:-30px;
    top:-30px;
    width:120px;
    height:120px;
    background:radial-gradient(circle, rgba(255,255,255,0.08), transparent 70%);
    pointer-events:none;
}

.app-tile:hover{
    transform:translateY(-4px);
    box-shadow:0 20px 36px rgba(0,0,0,0.34);
}

.tile-icon{
    font-size:34px;
    margin-bottom:12px;
}

.tile-title{
    font-size:22px;
    font-weight:800;
    margin-bottom:8px;
}

.tile-text{
    color:var(--muted);
    font-size:14px;
    line-height:1.55;
    margin-bottom:18px;
    flex-grow:1;
}

.attempt-card{
    background:rgba(255,255,255,0.05);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:12px;
    margin-bottom:14px;
}

.stat-card{
    background:
        linear-gradient(180deg, rgba(255,255,255,0.065), rgba(255,255,255,0.03));
    border:1px solid rgba(255,255,255,0.08);
    border-radius:22px;
    padding:18px;
    text-align:center;
    box-shadow:0 12px 24px rgba(0,0,0,0.22);
    margin-bottom:12px;
}

.stat-title{
    font-size:13px;
    color:var(--muted);
    margin-bottom:8px;
}

.stat-value{
    font-size:28px;
    font-weight:900;
}

.result-idle{
    background:linear-gradient(90deg,#374151,#4b5563);
    border-radius:22px;
    padding:24px;
    text-align:center;
    font-size:24px;
    font-weight:800;
    margin-top:10px;
}

.result-ok{
    background:linear-gradient(90deg,#00c96b,#00b95f);
    border-radius:22px;
    padding:24px;
    text-align:center;
    font-size:26px;
    font-weight:900;
    margin-top:10px;
    box-shadow:0 10px 24px rgba(0,201,107,0.25);
}

.result-bad{
    background:linear-gradient(90deg,#ff275f,#ff4d4d);
    border-radius:22px;
    padding:24px;
    text-align:center;
    font-size:26px;
    font-weight:900;
    margin-top:10px;
    box-shadow:0 10px 24px rgba(255,39,95,0.25);
}

.result-sub{
    font-size:15px;
    font-weight:500;
    opacity:0.95;
    margin-top:8px;
}

.metric{
    background:rgba(255,255,255,0.07);
    border-radius:18px;
    padding:16px;
    text-align:center;
    margin-bottom:12px;
}

.metric-title{
    font-size:13px;
    color:var(--muted);
    margin-bottom:8px;
}

.metric-value{
    font-size:28px;
    font-weight:800;
}

.info-box{
    background:rgba(255,255,255,0.06);
    border-left:4px solid #ff8a00;
    border-radius:14px;
    padding:14px;
    color:#f3f4f6;
    line-height:1.6;
    margin-top:12px;
}

.log-row{
    background:rgba(255,255,255,0.05);
    border-radius:14px;
    padding:10px 12px;
    margin-bottom:8px;
    border-left:4px solid #7b2ff7;
}

.stButton > button{
    width:100%;
    border:none;
    border-radius:14px;
    padding:0.85rem 1rem;
    font-weight:800;
    color:white;
    background:linear-gradient(90deg,#ff7a18,#af002d 42%,#7b2ff7);
    box-shadow:0 10px 22px rgba(123,47,247,0.22);
    transition:0.2s ease;
}

.stButton > button:hover{
    transform:translateY(-1px);
    opacity:0.96;
}

section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0d0f17 0%,#141826 100%);
    border-right:1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

.scan-line{
    height:4px;
    background:linear-gradient(90deg, transparent, #00ffd9, transparent);
    animation:scan 1s linear infinite;
    border-radius:999px;
    margin:10px 0 16px 0;
}

@keyframes scan{
    0%{transform:translateX(-100%)}
    100%{transform:translateX(100%)}
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TOP BAR
# =========================================================
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-dots">
            <div class="dot dot-red"></div>
            <div class="dot dot-yellow"></div>
            <div class="dot dot-green"></div>
        </div>
        <div>
            <div class="topbar-title">ACCESS TERMINAL</div>
            <div class="topbar-subtitle">Интеллектуальная система контроля доступа</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)



if not st.session_state.is_auth:
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Вход в систему</div>
        <div class="hero-subtitle">Авторизация администратора</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.2, 1])

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Авторизация")

        username = st.text_input("Логин")
        password = st.text_input("Пароль", type="password")

        if st.button("Войти"):
            if login_user(username, password):
                st.session_state.is_auth = True
                st.rerun()
            else:
                st.error("Неверный логин и пароль")

        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()
                   

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.markdown("## Навигация")
nav_button("Главная", "Главная", "🏠")
nav_button("Терминал", "Терминал", "🖥️")
nav_button("Регистрация", "Регистрация", "👤")
nav_button("Панель", "Панель", "📊")
nav_button("Журнал", "Журнал", "📝")
nav_button("Настройки", "Настройки", "⚙️")

st.sidebar.markdown("---")
st.sidebar.markdown("### Быстрый анализ")

if stats["total"] > 0:
    FAR = stats["false"] / stats["total"]
    FRR = stats["spoof"] / stats["total"]
    ACC = stats["true"] / stats["total"]
else:
    FAR = FRR = ACC = 0

st.sidebar.write(f"FAR: {FAR:.2f}")
st.sidebar.write(f"FRR: {FRR:.2f}")
st.sidebar.write(f"ACC: {ACC:.2f}")

if st.sidebar.button("Сброс системы", use_container_width=True):
    reset_all()
    go("Главная")

page = st.session_state.current_page

# =========================================================
# HOME
# =========================================================
if page == "Главная":
    st.markdown("""
    <div class="hero">
        <div class="hero-title">Добро пожаловать</div>
        <div class="hero-subtitle">Оборудование: DS-K1T342MX • DS-KH6110-WE1</div>
    </div>
    """, unsafe_allow_html=True)

    total = stats["total"]
    far = stats["false"] / total if total else 0
    frr = stats["spoof"] / total if total else 0
    acc = stats["true"] / total if total else 0
    security = get_security_level(far, frr)
    users_count = len(os.listdir(DB_PATH))

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Пользователи</div><div class="stat-value">{users_count}</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Попытки</div><div class="stat-value">{total}</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-card"><div class="stat-title">ACC</div><div class="stat-value">{acc:.2f}</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-card"><div class="stat-title">Безопасность</div><div class="stat-value">{security}</div></div>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("""
        <div class="app-tile">
            <div class="tile-icon">🖥️</div>
            <div class="tile-title">Терминал</div>
            <div class="tile-text">Проверка доступа по лицу через камеру, анализ подлинности и отображение результата в реальном времени.</div>
        </div>
        """, unsafe_allow_html=True)

    with t2:
        st.markdown("""
        <div class="app-tile">
            <div class="tile-icon">👤</div>
            <div class="tile-title">Регистрация</div>
            <div class="tile-text">Добавление пользователей, снимок через камеру, предпросмотр и управление базой эталонных лиц.</div>
        </div>
        """, unsafe_allow_html=True)

    with t3:
        st.markdown("""
        <div class="app-tile">
            <div class="tile-icon">📊</div>
            <div class="tile-title">Панель анализа</div>
            <div class="tile-text">Метрики точности, графики безопасности, уровень системы и общее состояние распознавания.</div>
        </div>
        """, unsafe_allow_html=True)

    b1, b2 = st.columns(2)
    with b1:
        st.markdown("""
        <div class="app-tile">
            <div class="tile-icon">📝</div>
            <div class="tile-title">Журнал</div>
            <div class="tile-text">Полная история событий: доступы, отказы, подозрительные изображения и время каждой попытки.</div>
        </div>
        """, unsafe_allow_html=True)

    with b2:
        st.markdown("""
        <div class="app-tile">
            <div class="tile-icon">⚙️</div>
            <div class="tile-title">Настройки</div>
            <div class="tile-text">Гибкая настройка чувствительности распознавания, анти-спуфинга и скорости сканирования.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("О системе")
    st.write("Система контроля и управления доступом с распознаванием лиц.")
    st.write("Что реализовано:")
    st.write("• распознавание лиц через камеру;")
    st.write("• FAR / FRR / ACC;")
    st.write("• анти-спуфинг;")
    st.write("• журнал событий;")
    st.write("• настройки безопасности.")
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TERMINAL
# =========================================================
elif page == "Терминал":
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown('<div class="terminal-panel">', unsafe_allow_html=True)
        st.subheader("Камера прохода")
        cam = st.camera_input("Сделайте снимок для проверки доступа")

        if cam is not None:
            img_preview = bytes_to_bgr(cam)
            if img_preview is not None:
                st.image(bgr_to_rgb(draw_face_box(img_preview)), use_container_width=True)

        if st.button("Сканировать доступ", key="scan_terminal"):
            if cam is None:
                st.warning("Сначала сделайте снимок камерой.")
            else:
                st.markdown('<div class="scan-line"></div>', unsafe_allow_html=True)

                progress = st.progress(0)
                for i in range(100):
                    time.sleep(settings["scan_delay"])
                    progress.progress(i + 1)

                img = bytes_to_bgr(cam)
                spoof_flag, variance, blur, _brightness = is_spoof(img)

                if not os.listdir(DB_PATH):
                    stats["false"] += 1
                    stats["total"] += 1
                    quality = get_quality_label(variance, blur)
                    set_result(
                        "bad",
                        "❌ ДОСТУП ЗАПРЕЩЕН",
                        "База пользователей пуста",
                        0,
                        variance,
                        blur,
                        "НИЗКАЯ",
                        quality
                    )
                    add_log("ОТКАЗ", "База пользователей пуста", 0)
                else:
                    best, diff = compare_faces(img)
                    similarity = max(0.0, min(100.0, 100 - diff))
                    confidence = get_confidence(similarity)
                    quality = get_quality_label(variance, blur)

                    stats["total"] += 1

                    if spoof_flag:
                        save_attempt_image(img, "spoof")
                        stats["spoof"] += 1
                        set_result(
                            "bad",
                            "⚠️ ПОДОЗРИТЕЛЬНОЕ ИЗОБРАЖЕНИЕ",
                            "Проверка подлинности не пройдена",
                            similarity,
                            variance,
                            blur,
                            confidence,
                            quality
                        )
                        add_log("СПУФИНГ", "Подозрительное изображение", similarity)

                    elif best and diff < settings["diff_threshold"] and similarity >= settings["similarity_threshold"]:
                        stats["true"] += 1
                        name = os.path.splitext(best)[0]
                        set_result(
                            "ok",
                            "✅ ДОСТУП РАЗРЕШЕН",
                            name,
                            similarity,
                            variance,
                            blur,
                            confidence,
                            quality
                        )
                        add_log("ДОСТУП", name, similarity)
                        st.balloons()

                    else:
                        save_attempt_image(img, "unknown")
                        stats["false"] += 1
                        set_result(
                            "bad",
                            "❌ ДОСТУП ЗАПРЕЩЕН",
                            "Неизвестный пользователь",
                            similarity,
                            variance,
                            blur,
                            confidence,
                            quality
                        )
                        add_log("ОТКАЗ", "Неизвестный пользователь", similarity)

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="terminal-panel">', unsafe_allow_html=True)
        st.subheader("Результат терминала")
        r = st.session_state.last_result

        if r["status"] == "ok":
            st.markdown(
                f'<div class="result-ok">{r["title"]}<div class="result-sub">{r["detail"]}</div></div>',
                unsafe_allow_html=True
            )
        elif r["status"] == "bad":
            st.markdown(
                f'<div class="result-bad">{r["title"]}<div class="result-sub">{r["detail"]}</div></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown('<div class="result-idle">ОЖИДАНИЕ</div>', unsafe_allow_html=True)

        if r["similarity"] is not None:
            st.markdown(
                f'''
                <div class="info-box">
                    <b>Сходство:</b> {r["similarity"]:.1f}%<br>
                    <b>Уверенность:</b> {r["confidence"]}<br>
                    <b>Качество:</b> {r["quality"]}<br>
                    <b>Variance:</b> {r["variance"]:.1f}<br>
                    <b>Blur:</b> {r["blur"]:.1f}
                </div>
                ''',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Последние события")
        if st.session_state.logs:
            for row in st.session_state.logs[:8]:
                st.markdown(
                    f'''
                    <div class="log-row">
                        <b>{row["time"]}</b> — {row["result"]}<br>
                        {row["detail"]}<br>
                        Сходство: {row["similarity"]}
                    </div>
                    ''',
                    unsafe_allow_html=True
                )
        else:
            st.info("Событий пока нет.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# REGISTRATION
# =========================================================
elif page == "Регистрация":
    left, right = st.columns([1.05, 1])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Регистрация пользователя")
        name = st.text_input("Имя пользователя")
        cam_reg = st.camera_input("Сделайте снимок для регистрации")

        if st.button("Сохранить пользователя", key="save_registration"):
            if not name:
                st.warning("Введите имя пользователя.")
            elif cam_reg is None:
                st.warning("Сначала сделайте снимок.")
            else:
                safe_name = name.strip().replace(" ", "_")
                path = os.path.join(DB_PATH, f"{safe_name}.jpg")
                with open(path, "wb") as f:
                    f.write(cam_reg.getvalue())
                st.success(f"Пользователь {safe_name} зарегистрирован.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Предпросмотр")
        if cam_reg is not None:
            img = bytes_to_bgr(cam_reg)
            if img is not None:
                st.image(bgr_to_rgb(draw_face_box(img)), use_container_width=True)
        else:
            st.info("После снимка с камеры здесь появится предпросмотр.")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("База пользователей")
    users = os.listdir(DB_PATH)
    st.write(f"Всего пользователей: {len(users)}")

    cols = st.columns(4)
    for i, user in enumerate(users):
        with cols[i % 4]:
            img = cv2.imread(os.path.join(DB_PATH, user))
            if img is not None:
                st.image(bgr_to_rgb(img), use_container_width=True)
                st.caption(user)

    if users:
        selected_user = st.selectbox("Удалить пользователя", users)
        if st.button("Удалить пользователя", key="delete_selected_user"):
            os.remove(os.path.join(DB_PATH, selected_user))
            st.rerun()

    if st.button("Очистить базу", key="clear_database"):
        for filename in users:
            os.remove(os.path.join(DB_PATH, filename))
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# PANEL
# =========================================================
elif page == "Панель":
    total = stats["total"]
    far = stats["false"] / total if total else 0
    frr = stats["spoof"] / total if total else 0
    acc = stats["true"] / total if total else 0
    security = get_security_level(far, frr)

    left, right = st.columns([1, 1.25])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Метрики")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric"><div class="metric-title">FAR</div><div class="metric-value">{far:.2f}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric"><div class="metric-title">FRR</div><div class="metric-value">{frr:.2f}</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric"><div class="metric-title">ACC</div><div class="metric-value">{acc:.2f}</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="metric"><div class="metric-title">Уровень безопасности</div><div class="metric-value">{security}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Графики")

        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        ax.bar(["FAR", "FRR", "ACC"], [far, frr, acc], color=["#ff7a18", "#ff0055", "#00c96b"], width=0.55)
        ax.set_title("Текущие показатели", color="white", fontsize=14, pad=14)
        ax.tick_params(colors="white", labelsize=11)
        ax.set_ylim(0, 1)
        for spine in ax.spines.values():
            spine.set_color("#888")
        ax.grid(axis="y", linestyle="--", alpha=0.18)    
        st.pyplot(fig, use_container_width=True)

        total_events = stats["true"] + stats["false"] + stats["spoof"]
        if total_events > 0:
            fig2, ax2 = plt.subplots(figsize=(7, 5))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")

            values = [stats["true"], stats["false"], stats["spoof"]]
            labels = ["Доступ", "Отказ", "Спуфинг"]
            colors = ["#00c96b", "#ff7a18", "#ff0055"]

            wedges, texts, autotexts = ax2.pie(
                values,
                labels=None,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"color": "white", "fontsize": 12}
            )

            for txt in autotexts:
                txt.set_color("white")
                txt.set_fontsize(12)
                txt.set_weight("bold")

            ax2.set_aspect("equal")
            ax2.legend(
                wedges,
                labels,
                loc="center",
                bbox_to_anchor=(0.5, -0.08),
                ncol=3,
                frameon=False,
                labelcolor="white",
                fontsize=11
            )

            st.pyplot(fig2, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)        

# =========================================================
# LOG
# =========================================================
elif page == "Журнал":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Полный журнал событий")

    if st.session_state.logs:
        for row in st.session_state.logs:
            st.markdown(
                f'''
                <div class="log-row">
                    <b>{row["time"]}</b> — {row["result"]}<br>
                    {row["detail"]}<br>
                    Сходство: {row["similarity"]}
                </div>
                ''',
                unsafe_allow_html=True
            )
    else:
        st.info("Журнал пока пуст.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Сохраненные попытки прохода")

    attempt_files = sorted(
        [f for f in os.listdir(ATTEMPTS_PATH) if f.lower().endswith((".jpg", ".jpeg", ".png"))],
        reverse=True
    )

    if attempt_files:
        cols = st.columns(3)
        for i, filename in enumerate(attempt_files[:18]):
            with cols[i % 3]:
                path = os.path.join(ATTEMPTS_PATH, filename)
                img = cv2.imread(path)
                if img is not None:
                    st.markdown('<div class="attempt-card">', unsafe_allow_html=True)
                    st.image(bgr_to_tgb(img), use_container_width=True)
                    st.caption(filename)
                    st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Сохраненных попыток пока нет.")

    st.markdown('</div>', unsafe_allow_html=True)                    

# =========================================================
# SETTINGS
# =========================================================
elif page == "Настройки":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Настройки системы")

    settings["diff_threshold"] = st.slider(
        "Порог diff для допуска",
        min_value=20,
        max_value=80,
        value=int(settings["diff_threshold"])
    )

    settings["similarity_threshold"] = st.slider(
        "Минимальное similarity для допуска",
        min_value=10,
        max_value=90,
        value=int(settings["similarity_threshold"])
    )

    settings["spoof_variance_threshold"] = st.slider(
        "Порог variance для анти-спуфинга",
        min_value=100,
        max_value=2000,
        value=int(settings["spoof_variance_threshold"])
    )

    settings["spoof_blur_threshold"] = st.slider(
        "Порог blur для анти-спуфинга",
        min_value=10,
        max_value=150,
        value=int(settings["spoof_blur_threshold"])
    )

    settings["scan_delay"] = st.slider(
        "Скорость анимации сканирования",
        min_value=0.001,
        max_value=0.03,
        value=float(settings["scan_delay"]),
        step=0.001
    )

    st.success("Параметры сохранены для текущей сессии.")
    st.markdown('</div>', unsafe_allow_html=True)