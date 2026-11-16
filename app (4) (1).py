import streamlit as st, sqlite3, json, re, random, io, glob, os, time, threading, html, hashlib, datetime as dt
from zoneinfo import ZoneInfo
from pypdf import PdfReader
from gtts import gTTS
import requests

st.set_page_config(page_title="Study Meta AI", page_icon="🎓", layout="centered")


def secret(name, default=""):
    # works on Streamlit Cloud (Secrets), locally (secrets.toml) and with plain environment variables
    try:
        v = st.secrets.get(name)
    except Exception:
        v = None
    return v or os.environ.get(name) or default


TEACHER_CODE = secret("TEACHER_CODE", "TEACH-2026")
GEMINI_KEY = secret("GEMINI_API_KEY")
MODEL = secret("MODEL", "gemini-3.8-flash")
TZ_NAME = secret("TIMEZONE", "Africa/Cairo")
try:
    TZ = ZoneInfo(TZ_NAME)
except Exception:
    TZ = dt.timezone(dt.timedelta(hours=2))


def now():
    return dt.datetime.now(TZ)


def today():
    return now().date()


YEARS = ["Secondary 1", "Secondary 2", "Secondary 3"]
MOT = ["You're doing great! Keep going 💪", "Every question makes you stronger 🌟",
       "Small steps every day lead to big results 🚀", "Believe in yourself, you've got this! ✨",
       "Practice makes progress 📈", "Great focus! Stay curious 🔍"]
ss = st.session_state


# ---------------------------------------------------------------- database
@st.cache_resource
def db():
    c = sqlite3.connect("studymeta.db", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript("""
    create table if not exists users(id integer primary key autoincrement, name text, role text, year int, subject text);
    create table if not exists lessons(id integer primary key autoincrement, title text, text text, year int);
    create table if not exists results(id integer primary key autoincrement, sid int, lesson text, kind text, score int, total int, ts text);
    create table if not exists msgs(id integer primary key autoincrement, frm text, to_id text, text text, ts text);
    create table if not exists plans(sid int primary key, data text);
    create table if not exists aicache(k text primary key, v text);""")
    c.commit()
    return c


@st.cache_resource
def _lock():
    return threading.RLock()


def q(sql, args=(), one=False):
    with _lock():
        cur = db().execute(sql, args)
        rows = [dict(r) for r in cur.fetchall()]
        db().commit()
    return (rows[0] if rows else None) if one else rows


def ins(sql, args=()):
    with _lock():
        cur = db().execute(sql, args)
        db().commit()
        return cur.lastrowid


def clean(t):
    return (t or "").replace("\x00", "").strip()


def pdf_text(f):
    return clean("\n\n".join((p.extract_text() or "") for p in PdfReader(f).pages))


def read_upload(f):
    try:
        if f.name.lower().endswith(".pdf"):
            return pdf_text(f)
        return clean(f.read().decode("utf-8", "ignore"))
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        return ""


def add_lesson(title, text, year):
    # saving the same lesson twice must not create a second copy in the library
    if q("select id from lessons where title=? and text=? and year=?", (title, text, year), one=True):
        return
    q("insert into lessons(title,text,year) values(?,?,?)", (title, text, year))


def library(year):
    seen, out = set(), []
    for l in q("select * from lessons where year=? or year=-1 order by id", (year,)):
        key = (l["title"].strip().lower(), hash(l["text"]))
        if key not in seen:
            seen.add(key)
            out.append(l)
    return out


def seed():
    if q("select count(*) n from lessons", one=True)["n"]:
        return
    for p in glob.glob("seed/*.pdf"):
        try:
            t = pdf_text(p)
        except Exception:
            continue
        if len(t) > 100:
            q("insert into lessons(title,text,year) values(?,?,?)", (os.path.basename(p)[:-4].replace("_", " "), t, -1))


seed()


# ---------------------------------------------------------------- AI helpers
RETRY_CODES = (429, 500, 502, 503, 504)


@st.cache_resource
def _model_state():
    return {"ok": None}  # remembers the model that last worked


BUSY_MSG = "The AI is busy right now (free-tier limit). Wait a minute and try again. | الذكاء الاصطناعي مشغول، استني دقيقة وجرّبي تاني."
DAILY_MSG = "Today's free AI limit has finished for this key. Try again tomorrow or use a new API key. | الحد المجاني لليوم خلص."


def model_list():
    # the configured model first; each model has its own free quota, so if one is used up the next one is tried
    ms = [_model_state()["ok"], MODEL, "gemini-3.8-flash", "gemini-flash-latest", "gemini-flash-lite-latest"]
    return [m for i, m in enumerate(ms) if m and m not in ms[:i]]


def gen(contents, max_tokens=4000, as_json=False):
    if not GEMINI_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY in the app Secrets.")
    missing = ""
    busy = daily = False
    last = ""
    for model in model_list():
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        cfg = {"maxOutputTokens": max(max_tokens, 8000)}
        if as_json:
            cfg["responseMimeType"] = "application/json"
        if model.startswith("gemini-2.5-flash"):
            cfg["thinkingConfig"] = {"thinkingBudget": 0}  # faster answers, no output budget lost to "thinking"
        for attempt in range(3):
            try:
                r = requests.post(url, headers={"x-goog-api-key": GEMINI_KEY},
                                  json={"contents": contents, "generationConfig": cfg}, timeout=120)
            except requests.RequestException:
                time.sleep(3 * (attempt + 1))
                continue
            if r.status_code == 404:  # model retired or not available to this key: try the next one
                missing = r.text[:200]
                break
            last = f"{r.status_code} {r.text[:160]}"
            if r.status_code == 429 and "PerDay" in r.text:  # daily quota of this model is finished: waiting will not help
                daily = True
                break
            if r.status_code in RETRY_CODES:
                busy = True
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"AI error {r.status_code}: {r.text[:200]}")
            try:
                text = "".join(p.get("text", "") for p in r.json()["candidates"][0]["content"]["parts"])
            except Exception:
                text = ""
            if not text.strip():
                raise RuntimeError("The AI returned an empty answer. Please try again.")
            _model_state()["ok"] = model
            return text
    if busy:
        raise RuntimeError(BUSY_MSG + "  [" + last + "]")
    if daily:
        raise RuntimeError(DAILY_MSG + "  [" + last + "]")
    if missing:
        raise RuntimeError("No Gemini model is available for this key. Set MODEL in the app Secrets to a current model name. " + missing)
    raise RuntimeError("Could not reach the AI service. Check the internet connection and try again.")


def cached(kind, text, make, force=False):
    # saved answers: the same lesson text never calls the AI twice (kept in the database)
    k = kind + ":" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    if not force:
        row = q("select v from aicache where k=?", (k,), one=True)
        if row:
            try:
                return json.loads(row["v"])
            except Exception:
                pass
    val = make()
    ins("insert or replace into aicache(k, v) values(?, ?)", (k, json.dumps(val, ensure_ascii=False)))
    return val


def ask(prompt, fast=False, max_tokens=4000, messages=None):
    if messages:
        contents = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [{"text": m["content"]}]} for m in messages]
    else:
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
    return gen(contents, max_tokens)


def ask_json(prompt, fast=False, max_tokens=8000):
    t = gen([{"role": "user", "parts": [{"text": prompt}]}], max_tokens, True)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", t, re.S)
        if not m:
            raise RuntimeError("The AI answer was not in the expected format. Please try again.")
        try:
            return json.loads(m.group(0))
        except Exception:
            raise RuntimeError("The AI answer was cut off or malformed. Please try again.")


def as_list(x):
    if isinstance(x, dict):
        for v in x.values():
            if isinstance(v, list):
                return v
        return []
    return x if isinstance(x, list) else []


def to_int(v, d=30):
    try:
        return int(v)
    except Exception:
        return d


# ---------------------------------------------------------------- text helpers
def sections(t):
    merged = {}  # the same lesson title can show up many times (page headers, repeated headings): keep each lesson once
    for p in re.split(r"(?m)^(?=Lesson \d+\s*:)", t):
        m = re.match(r"Lesson (\d+)\s*:\s*([^\n=.]+)", p)
        title = f"Lesson {m.group(1)}: {m.group(2).strip()}" if m else "Overview"
        if len(p.strip()) > 80:
            key = re.sub(r"\s+", " ", title).strip().lower()
            merged.setdefault(key, [title, []])[1].append(p.strip())
    return [(v[0], "\n\n".join(v[1])) for v in merged.values()]


def chunks(t, n=700):
    out, cur = [], ""
    for p in re.split(r"\n\s*\n|\n", t):
        if len(cur) + len(p) > n and cur:
            out.append(cur)
            cur = ""
        cur += p + "\n"
    return out + ([cur] if cur.strip() else [])


def retrieve(question, t, k=4):
    w = [x for x in re.split(r"\W+", question.lower()) if len(x) > 2]
    sc = [(sum(c.lower().count(x) for x in w), i + 1, c) for i, c in enumerate(chunks(t))]
    best = sorted(sc, key=lambda x: (-x[0], x[1]))[:k]
    if not best or best[0][0] == 0:  # e.g. Arabic question about an English lesson: fall back to the start of the lesson
        best = sc[:k]
    return best


def stats(uid):
    r = q("select * from results where sid=?", (uid,))
    days = {x["ts"][:10] for x in r}
    d = today()
    if d.isoformat() not in days:
        d -= dt.timedelta(days=1)
    streak = 0
    while d.isoformat() in days:
        streak += 1
        d -= dt.timedelta(days=1)
    return sum(x["score"] * 10 for x in r), streak


def praise(p):
    return "Outstanding work! 🏆" if p >= 85 else "Good job! Review the mistakes and you'll be even better 💪" if p >= 60 \
        else "Don't give up! Review the lesson and try again, you will improve 🌱"


# ---------------------------------------------------------------- auth
def welcome():
    st.title("🎓 Study Meta AI")
    st.caption("Smart learning platform: students, teachers and AI in one system")
    role = st.radio("Who are you?", ["Student", "Teacher"], horizontal=True)
    if role == "Student":
        t1, t2 = st.tabs(["New student", "Returning student"])
        with t1:
            name = st.text_input("Your name")
            yr = st.selectbox("Year", range(3), format_func=lambda i: YEARS[i])
            if st.button("Sign up", type="primary") and name.strip():
                uid = ins("insert into users(name,role,year) values(?,?,?)", (name.strip(), "student", yr))
                ss.user = dict(q("select * from users where id=?", (uid,), one=True))
                st.rerun()
        with t2:
            ex = q("select * from users where role='student'")
            if ex:
                pick = st.selectbox("Choose your name", ex, format_func=lambda r: f"{r['name']} ({YEARS[r['year']]})")
                if st.button("Log in"):
                    ss.user = dict(pick)
                    st.rerun()
            else:
                st.info("No students yet.")
    else:
        name = st.text_input("Teacher name")
        subj = st.text_input("Subject")
        code = st.text_input("Teachers code", type="password")
        if st.button("Log in", type="primary"):
            if code != TEACHER_CODE:
                st.error("Incorrect teachers code")
            elif name.strip():
                r = q("select * from users where role='teacher' and name=? and subject=?", (name.strip(), subj.strip()), one=True)
                if not r:
                    uid = ins("insert into users(name,role,subject) values(?,?,?)", (name.strip(), "teacher", subj.strip() or "General"))
                    r = q("select * from users where id=?", (uid,), one=True)
                ss.user = dict(r)
                st.rerun()


# ---------------------------------------------------------------- student
STEPS = ["Upload", "Explain", "Summary", "Quiz", "Final Exam", "Score"]
WORK = ["exp", "expAr", "sum", "quiz", "exam", "audio", "chat"]


def clear_work():
    # everything generated for the previous lesson (explanation, summary, quizzes, chat, answers)
    for k in WORK:
        ss.pop(k, None)
    for k in [k for k in list(ss.keys()) if re.match(r"(quiz|exam)_\d+$", str(k))]:
        del ss[k]


def reset_lesson():
    for k in ["doc", "secs", "pick", "lesson"]:
        ss.pop(k, None)
    clear_work()
    ss.step = 0


def begin(title, text):
    s = sections(text)
    reset_lesson()
    if len(s) > 1:
        ss.doc, ss.secs, ss.pick = (title, text), s, True
    else:
        ss.lesson = (title, text)
        ss.step = 1


def step_upload(u):
    if ss.get("pick"):
        st.subheader(f"📘 {ss.doc[0]}")
        st.caption("Choose the lesson to study. Explanation, summary, quiz, exam and chat will focus on it.")
        for i, (t, body) in enumerate(ss.secs):
            if st.button(f"📖 {t}", key=f"s_{i}", use_container_width=True):
                clear_work()
                ss.lesson = (f"{ss.doc[0][:30]} – {t}", body)
                ss.pick = False
                ss.step = 1
                st.rerun()
        if st.button("📚 Whole unit", use_container_width=True):
            clear_work()
            ss.lesson = ss.doc
            ss.pick = False
            ss.step = 1
            st.rerun()
        return
    st.subheader("Upload a lesson")
    f = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    title = st.text_input("Lesson title")
    text = st.text_area("…or paste the text", height=120)
    if st.button("Use this lesson", type="primary"):
        t = text
        if f is not None:
            t = read_upload(f)
            title = title or f.name.rsplit(".", 1)[0]
        t = clean(t)
        if len(t) < 40:
            st.error("No readable text found (scanned PDF? paste the text instead).")
        else:
            add_lesson(title or "New lesson", t, u["year"])
            begin(title or "New lesson", t)
            st.rerun()
    st.subheader("Library")
    for l in library(u["year"]):
        if st.button(f"📘 {l['title']}", key=f"l{l['id']}", use_container_width=True):
            begin(l["title"], l["text"])
            st.rerun()


def nxt(label="Next ➡"):
    if st.button(label, type="primary"):
        ss.step += 1
        st.rerun()


def step_explain():
    t, body = ss.lesson
    st.subheader(f"📘 {t}")
    if st.button("↩ Choose another lesson"):
        clear_work()
        if ss.get("secs"):
            ss.pick = True
            ss.step = 0
            ss.pop("lesson", None)
        else:
            reset_lesson()
        st.rerun()
    if st.button("Explain the lesson" if not ss.get("exp") else "Explain again"):
        with st.spinner("Explaining…"):
            try:
                _p = f"Explain the following lesson to a secondary school student in clear, simple English. Use short sections, examples and analogies. Stay within the lesson content.\n\n{body[:12000]}"
                ss.exp = cached("exp", body[:12000], lambda: ask(_p), force=bool(ss.get("exp")))
                ss.pop("expAr", None)
            except Exception as ex:
                st.error(str(ex))
    if ss.get("exp"):
        st.markdown(ss.exp)
        if st.button("🌐 Translate to Arabic"):
            with st.spinner("Translating…"):
                try:
                    ss.expAr = cached("expAr", ss.exp, lambda: ask("Translate the following explanation into clear Arabic. Keep key English technical terms in parentheses.\n\n" + ss.exp))
                except Exception as ex:
                    st.error(str(ex))
        if ss.get("expAr"):
            st.markdown(f"<div dir='rtl'>\n\n{ss.expAr}\n\n</div>", unsafe_allow_html=True)
        nxt()


def tts(text, lang):
    buf = io.BytesIO()
    gTTS(text=text[:4500], lang=lang).write_to_fp(buf)
    return buf.getvalue()


def step_summary():
    t, body = ss.lesson
    if st.button("Summarize the lesson"):
        with st.spinner("Summarizing…"):
            try:
                def _mk():
                    r = ask_json(f'Summarize the lesson below. Return JSON only: {{"en":{{"summary":"...","points":["..."],"terms":[{{"en":"term","def":"short definition"}}]}},"ar":{{"summary":"...","points":["..."]}}}}. 5 to 7 short key points covering definitions, rules and formulas. "ar" is the Arabic translation of the English summary and points.\n\nLesson:\n{body[:12000]}')
                    if not (isinstance(r, dict) and isinstance(r.get("en"), dict)):
                        raise RuntimeError("Unexpected AI format. Please try again.")
                    return r
                r = cached("sum", body[:12000], _mk)
                ss.sum = r
                ss.audio = {}
            except Exception as ex:
                st.error(str(ex))
    S = ss.get("sum")
    if not S:
        return
    ss.setdefault("audio", {})
    E, A = S.get("en") or {}, S.get("ar") or {}
    st.markdown("### Summary")
    st.write(E.get("summary", ""))
    for p in E.get("points", []):
        st.markdown(f"- {p}")
    terms = [x for x in E.get("terms", []) if isinstance(x, dict)]
    if terms:
        st.table([{"Term": x.get("en", ""), "Definition": x.get("def", "")} for x in terms])
    if st.checkbox("Show Arabic"):
        lis = "".join(f"<li>{html.escape(str(p))}</li>" for p in A.get("points", []))
        summ = html.escape(str(A.get("summary", "")))
        st.markdown(f"<div dir='rtl'><h3>الملخص</h3><p>{summ}</p><ul>{lis}</ul></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    for col, lang, d, lab in [(c1, "en", E, "🔊 English audio"), (c2, "ar", A, "🔊 Arabic audio")]:
        if col.button(lab, use_container_width=True):
            with st.spinner("Generating audio…"):
                try:
                    parts = ([t] if lang == "en" else []) + [str(d.get("summary", ""))] + [str(p) for p in d.get("points", [])]
                    ss.audio[lang] = tts(" . ".join(parts), lang)
                except Exception as ex:
                    st.error(f"Audio failed: {ex}")
    for lang in ("en", "ar"):
        if ss.audio.get(lang):
            st.caption("English narration" if lang == "en" else "التعليق الصوتي بالعربي")
            st.audio(ss.audio[lang], format="audio/mp3")
    nxt()


def norm(x):
    if not isinstance(x, dict):
        return None
    o = x.get("options")
    try:
        a = int(x.get("answer"))
    except Exception:
        return None
    if not x.get("q") or not isinstance(o, list) or len(o) < 2 or not 0 <= a < len(o):
        return None
    return {"q": str(x["q"]), "options": [str(t) for t in o], "answer": a, "explain": str(x.get("explain", ""))}


def qprompt(n, extra, body):
    return (f"Create {n} exam questions in English from the lesson below only (about 70% multiple choice with 4 options, 30% true/false). {extra} "
            'Return JSON only: an array of {"q":"question","options":["..."],"answer":index of correct option starting at 0,"explain":"short explanation"}. '
            f'For true/false use options ["True","False"].\n\nLesson:\n{body[:12000]}')


def make_questions(body, n):
    per = 10
    N = max(1, n // per)
    out, seen = [], []
    bar = st.progress(0.0, text="Generating questions…")
    k, errs = 1, 0
    while len(out) < n and k <= N + 3:
        part = (k - 1) % N + 1
        extra = f"Focus on part {part} of {N} of the lesson (split by position) and vary the difficulty. Do NOT repeat: {' | '.join(seen[-25:])}"
        try:
            for raw in as_list(ask_json(qprompt(per, extra, body), fast=True)):
                x = norm(raw)
                if x and x["q"] not in seen:
                    seen.append(x["q"])
                    out.append(x)
        except Exception as ex:
            st.warning(str(ex))
            errs += 1
            if not GEMINI_KEY or errs >= 3:
                break
            time.sleep(3)
        bar.progress(min(1.0, len(out) / n), text=f"Generating {len(out)}/{n}…")
        k += 1
    bar.empty()
    qs = []
    for x in out[:n]:
        o = [(t, i == x["answer"]) for i, t in enumerate(x["options"])]
        if len(o) > 2:
            random.shuffle(o)
        qs.append({"q": x["q"], "o": o, "e": x["explain"]})
    return qs


def quiz_block(key, n, kind, body, title, selector=False):
    if selector:
        n = st.selectbox("Number of questions", [40, 60])
    if st.button("Start" if not ss.get(key) else "Restart", type="primary"):
        qs = make_questions(body, n)
        if not qs:
            st.error("Could not generate questions. Please try again in a minute.")
        else:
            for k in [k for k in list(ss.keys()) if re.match(rf"{key}_\d+$", str(k))]:
                del ss[k]
            ss[key] = {"qs": qs, "done": False}
            st.rerun()
    Q = ss.get(key)
    if not Q or not Q["qs"]:
        return
    if not Q["done"]:
        with st.form(key + "_form"):
            for i, x in enumerate(Q["qs"]):
                st.radio(f"{i+1}. {x['q']}", [t for t, _ in x["o"]], index=None, key=f"{key}_{i}")
            if st.form_submit_button("Submit and grade", type="primary"):
                sc = sum(1 for i, x in enumerate(Q["qs"]) if any(c and t == ss.get(f"{key}_{i}") for t, c in x["o"]))
                Q["done"], Q["score"] = True, sc
                q("insert into results(sid,lesson,kind,score,total,ts) values(?,?,?,?,?,?)",
                  (ss.user["id"], title, kind, sc, len(Q["qs"]), now().isoformat()))
                st.rerun()
    else:
        for i, x in enumerate(Q["qs"]):
            mine = ss.get(f"{key}_{i}")
            st.markdown(f"**{i+1}. {x['q']}**")
            for t, c in x["o"]:
                st.markdown(("✅ " if c else "❌ " if t == mine else "▫️ ") + t)
            st.caption("💡 " + x["e"])
        p = round(100 * Q["score"] / len(Q["qs"]))
        st.success(f"Your score: {Q['score']} / {len(Q['qs'])} ({p}%)  {praise(p)}")
        nxt()


def step_score():
    s = t = 0
    for k in ("quiz", "exam"):
        if ss.get(k) and ss[k].get("done"):
            s += ss[k]["score"]
            t += len(ss[k]["qs"])
    p = round(100 * s / t) if t else 0
    st.markdown(f"## {'🏆' if p >= 60 else '📚'} {s} / {t} ({p}%)")
    st.write(praise(p))
    if st.button("New lesson", type="primary"):
        reset_lesson()
        st.rerun()


def student(u):
    ss.setdefault("step", 0)
    page = st.sidebar.radio("Menu", ["📖 Study", "🤖 AI Chat", "📅 Study Plan", "📩 Inbox"])
    pts, days = stats(u["id"])
    lv = "Master 🏅" if pts >= 500 else "Achiever 🥈" if pts >= 150 else "Beginner 🌱"
    st.info(f"{MOT[(ss.step + pts // 10) % len(MOT)]}   ·   ⭐ {pts} pts   ·   🔥 {days}-day streak   ·   {lv}")
    if page == "🤖 AI Chat":
        return chat_page()
    if page == "📅 Study Plan":
        return plan_page(u)
    if page == "📩 Inbox":
        return inbox_page(u)
    st.progress(ss.step / (len(STEPS) - 1))
    st.caption("  →  ".join(f"**{s}**" if i == ss.step else s for i, s in enumerate(STEPS)))
    if ss.step == 0:
        return step_upload(u)
    if "lesson" not in ss:
        reset_lesson()
        return st.rerun()
    t, body = ss.lesson
    if ss.step == 1:
        step_explain()
    elif ss.step == 2:
        step_summary()
    elif ss.step == 3:
        st.subheader("Quick quiz (10 questions)")
        quiz_block("quiz", 10, "quiz", body, t)
    elif ss.step == 4:
        st.subheader("Comprehensive final exam")
        quiz_block("exam", 40, "exam", body, t, selector=True)
    else:
        step_score()


def chat_page():
    st.subheader("🤖 AI Tutor")
    if "lesson" not in ss:
        return st.warning("Choose a lesson in the Study page first.")
    t, body = ss.lesson
    st.caption(f"Answering only from: {t}")
    lang = st.radio("Answer language", ["English", "Arabic"], horizontal=True)
    ss.setdefault("chat", [])
    for m in ss.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
    if msg := st.chat_input("Ask a question about the lesson…"):
        ctx = "\n".join(f"[Passage {i}] {c}" for _, i, c in retrieve(msg, body))
        sysm = (f"You are a smart tutor. Answer ONLY from the lesson passages below. If the answer is not in them, say clearly that it is not in the lesson and do not invent anything. Cite the passage number. Explain simply. Language: {lang}.\n\nPassages:\n{ctx}\n\n")
        prev = ss.chat[-6:]
        while prev and prev[0]["role"] != "user":
            prev = prev[1:]
        msgs = prev + [{"role": "user", "content": sysm + "Question: " + msg}]
        ss.chat.append({"role": "user", "content": msg})
        with st.chat_message("user"):
            st.markdown(msg)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    a = ask("", messages=msgs)
                except Exception as e:
                    a = None
                    st.error("⚠️ " + str(e))
            if a:
                st.markdown(a)
        if a:
            ss.chat.append({"role": "assistant", "content": a})
        else:
            ss.chat.pop()  # keep the history alternating user/assistant


def topics(u):
    out = []
    for l in library(u["year"]):
        s = sections(l["text"])
        out += [f"{l['title'][:30]} – {t}" for t, _ in s] if len(s) > 1 else [l["title"]]
    return list(dict.fromkeys(out))


def plan_page(u):
    st.subheader("📅 Smart Study Plan")
    row = q("select data from plans where sid=?", (u["id"],), one=True)
    P = json.loads(row["data"]) if row else None
    tp = topics(u)
    with st.form("planf"):
        d = st.date_input("Exam date", value=today() + dt.timedelta(days=7), min_value=today() + dt.timedelta(days=1))
        h = st.selectbox("Study hours per day", [1, 2, 3, 4], index=1)
        sel = st.multiselect("Topics", tp, default=tp)
        resched = st.checkbox("Reschedule unfinished tasks from my current plan", value=True)
        go = st.form_submit_button("Generate plan", type="primary")
    if go and not sel:
        st.warning("Choose at least one topic.")
    if go and sel:
        weak = list({r["lesson"] for r in q("select * from results where sid=?", (u["id"],)) if r["total"] and r["score"] / r["total"] < .6})
        left = [t["t"] for day in P["days"] for t in day["tasks"] if not t["done"]][:20] if (P and resched) else []
        joined = "; ".join(sel)
        weak_s = "; ".join(weak) or "none"
        left_s = ("Unfinished tasks to reschedule first: " + "; ".join(left) + ".") if left else ""
        prompt = (f"Today is {today().isoformat()}. The exam is on {d}. The student can study {h} hours per day. Topics: {joined}. Weak topics to prioritize and repeat: {weak_s}. "
                  f"{left_s} Create a day-by-day study plan from today until the day before the exam (max 14 days). Harder or weak topics earlier, short revision, and a mock exam on the last day. "
                  f'Return JSON only: an array of {{"date":"YYYY-MM-DD","tasks":[{{"t":"task in English","min":45}}]}}. Max {h*60} minutes per day.')
        saved = False
        with st.spinner("Planning…"):
            try:
                days = []
                for x in as_list(ask_json(prompt, fast=True)):
                    if not isinstance(x, dict) or not x.get("date"):
                        continue
                    tasks = [{"t": str(t["t"]), "min": to_int(t.get("min", 30)), "done": False}
                             for t in x.get("tasks", []) if isinstance(t, dict) and t.get("t")]
                    if tasks:
                        days.append({"date": str(x["date"])[:10], "tasks": tasks})
                if not days:
                    raise RuntimeError("The AI returned an empty plan. Please try again.")
                days.sort(key=lambda z: z["date"])
                P = {"exam": str(d), "days": days}
                q("insert or replace into plans(sid,data) values(?,?)", (u["id"], json.dumps(P)))
                for k in [k for k in list(ss.keys()) if re.match(r"t\d+_\d+$", str(k))]:
                    del ss[k]  # old checkbox states must not leak into the new plan
                saved = True
            except Exception as ex:
                st.error(str(ex))
        if saved:
            st.rerun()
    if not P:
        return
    allt = [t for day in P["days"] for t in day["tasks"]]
    dn = sum(t["done"] for t in allt)
    st.progress(dn / max(1, len(allt)), text=f"Progress: {dn}/{len(allt)} tasks")
    td = today().isoformat()
    for i, day in enumerate(P["days"]):
        st.markdown(f"**{'📍 Today · ' if day['date'] == td else ''}{day['date']}**")
        for j, t in enumerate(day["tasks"]):
            late = " ⏰ overdue" if (not t["done"] and day["date"] < td) else ""

            def tick(i=i, j=j):
                P["days"][i]["tasks"][j]["done"] = ss[f"t{i}_{j}"]
                q("insert or replace into plans(sid,data) values(?,?)", (u["id"], json.dumps(P)))
            st.checkbox(f"{t['t']} ({t['min']} min){late}", value=t["done"], key=f"t{i}_{j}", on_change=tick)


def inbox_page(u):
    st.subheader("📩 Messages from teachers")
    ms = q("select * from msgs where to_id='all' or to_id=? order by id desc", (str(u["id"]),))
    for m in ms:
        st.info(f"**{m['frm']}**{' (all students)' if m['to_id']=='all' else ''}\n\n{m['text']}")
    if not ms:
        st.caption("No messages yet.")


# ---------------------------------------------------------------- teacher
def teacher(u):
    st.subheader(f"👩‍🏫 {u['name']} · {u['subject']}")
    t1, t2, t3 = st.tabs(["👥 Students", "💬 Messages", "📤 Upload lesson"])
    studs = q("select * from users where role='student'")
    with t1:
        rows = []
        for s in studs:
            r = q("select * from results where sid=?", (s["id"],))
            tot = sum(x["total"] for x in r)
            avg = round(100 * sum(x["score"] for x in r) / tot) if tot else None
            weak = sorted({x["lesson"] for x in r if x["total"] and x["score"] / x["total"] < .6})
            rows.append({"Name": s["name"], "Year": YEARS[s["year"]], "Average": f"{avg}%" if avg is not None else "—", "Weak in": ", ".join(weak) or "—"})
        if rows:
            st.table(rows)
        else:
            st.caption("No students yet.")
        res = q("select r.*, u.name from results r join users u on u.id=r.sid order by r.id desc limit 15")
        if res:
            st.markdown("**Latest results**")
            st.table([{"Student": x["name"], "Lesson": x["lesson"], "Type": "Final exam" if x["kind"] == "exam" else "Quiz", "Score": f"{x['score']}/{x['total']}"} for x in res])
    with t2:
        if not studs:
            st.caption("No students to message yet.")
        else:
            to = st.selectbox("To", ["all"] + [str(s["id"]) for s in studs],
                              format_func=lambda i: "All students" if i == "all" else next(s["name"] for s in studs if str(s["id"]) == i))
            txt = st.text_area("Message or feedback")
            if st.button("Send", type="primary") and txt.strip():
                q("insert into msgs(frm,to_id,text,ts) values(?,?,?,?)", (f"{u['name']} ({u['subject']})", to, txt.strip(), now().isoformat()))
                st.success("Sent")
    with t3:
        yr = st.selectbox("Year", [-1, 0, 1, 2], format_func=lambda i: "All years" if i < 0 else YEARS[i])
        f = st.file_uploader("PDF or TXT", type=["pdf", "txt"], key="tu")
        title = st.text_input("Title", key="tt")
        text = st.text_area("…or paste text", key="tx")
        if st.button("Save to library"):
            t = text
            if f is not None:
                t = read_upload(f)
                title = title or f.name.rsplit(".", 1)[0]
            t = clean(t)
            if len(t) < 40:
                st.error("No readable text.")
            else:
                add_lesson(title or "New lesson", t, yr)
                st.success("Saved to the library")


# ---------------------------------------------------------------- main
if "user" not in ss:
    welcome()
else:
    u = ss.user
    with st.sidebar:
        st.write(f"**{u['name']}**" + (f" · {YEARS[u['year']]}" if u["role"] == "student" else ""))
        if st.button("Log out"):
            ss.clear()
            st.rerun()
    if u["role"] == "student":
        student(u)
    else:
        teacher(u)
