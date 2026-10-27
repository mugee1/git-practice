"""AI 면접 코치 — Streamlit 화면만 둡니다."""

import streamlit as st

import coach
from llm import LLMError, ask_json
from prompts import INTERVIEWER_SCHEMA

MIN_EXPERIENCE_LEN = 30

st.set_page_config(page_title="AI 면접 코치", page_icon="🎤")


def reset():
    st.session_state.clear()
    st.session_state.stage = "start"


if "stage" not in st.session_state:
    reset()
ss = st.session_state


# ---------- 사이드바 상태창 ----------
with st.sidebar:
    st.header("상태창")
    if ss.stage != "start":
        current = min(ss.q_index + 1, coach.TOTAL_QUESTIONS)
        st.progress(ss.q_index / coach.TOTAL_QUESTIONS, text=f"질문 {current}/{coach.TOTAL_QUESTIONS}")
        understanding = ss.understanding
        st.metric("면접관이 본 이해도", "-" if understanding is None else f"{understanding} / 100")
    if st.button("처음부터 다시", use_container_width=True):
        reset()
        st.rerun()


# ---------- 시작 화면 ----------
def start_screen():
    st.title("🎤 AI 면접 코치")
    st.caption("클래스로 신입 모의면접 · 질문 5개")
    with st.form("start"):
        name = st.text_input("이름")
        job = st.selectbox("직군", coach.JOBS)
        company = st.text_input("회사", value="클래스로")
        experience = st.text_area("내 경험", placeholder="프로젝트 · 맡은 일 · 결과를 적어 주세요 (30자 이상)")
        submitted = st.form_submit_button("면접 시작")

    if not submitted:
        return
    if not name.strip() or not company.strip():
        st.warning("이름과 회사를 입력해 주세요.")
        return
    if len(experience.strip()) < MIN_EXPERIENCE_LEN:
        st.warning(f"경험을 {MIN_EXPERIENCE_LEN}자 이상 적어 주세요. (지금 {len(experience.strip())}자)")
        return

    try:
        questions = coach.pick_questions(coach.load_questions(), job)
    except (OSError, ValueError):
        st.error("질문 은행(data/questions.csv)을 읽지 못했습니다.")
        return

    ss.update(
        stage="interview",
        name=name.strip(),
        job=job,
        company=company.strip(),
        experience=experience.strip(),
        questions=questions,
        q_index=0,
        follow_ups_used=0,
        understanding=None,
        instructions=coach.build_instructions(job, company.strip(), experience.strip()),
        messages=[{
            "role": "assistant",
            "content": "면접을 시작하겠습니다.\n\n" + coach.main_question_text(0, questions),
        }],
    )
    st.rerun()


# ---------- 면접 화면 ----------
def finish(message: str):
    ss.messages.append({"role": "assistant", "content": message})
    ss.stage = "done"


def handle_answer(answer: str) -> bool:
    """성공하면 True. 에러면 화면에 한 줄 띄우고 False."""
    if answer.strip() == "종료":
        ss.messages.append({"role": "user", "content": answer})
        finish("면접을 종료합니다. 수고하셨습니다.")
        return True

    history = ss.messages + [{"role": "user", "content": answer}]
    turn_input = coach.build_turn_input(ss.questions, ss.q_index, ss.follow_ups_used, history)
    with st.spinner("면접관이 답변을 보고 있어요..."):
        try:
            result = ask_json(ss.instructions, turn_input, INTERVIEWER_SCHEMA, "interviewer")
        except LLMError as e:
            st.error(str(e))
            return False

    ss.messages.append({"role": "user", "content": answer})
    ss.understanding = coach.clamp_understanding(result.get("understanding"))

    if coach.can_follow_up(result.get("follow_up"), ss.follow_ups_used):
        ss.follow_ups_used += 1
        ss.messages.append({"role": "assistant", "content": result.get("reply", "")})
        return True

    # 꼬리 질문이 없거나 이미 1번 했으면 코드가 다음 메인 질문으로 넘깁니다
    ss.q_index += 1
    ss.follow_ups_used = 0
    if ss.q_index >= coach.TOTAL_QUESTIONS:
        finish("메인 질문 5개를 모두 마쳤습니다. 수고하셨습니다.")
    else:
        ss.messages.append({"role": "assistant", "content": coach.main_question_text(ss.q_index, ss.questions)})
    return True


def interview_screen():
    st.title("🎤 모의면접")
    st.caption(f"{ss.name} · {ss.company} {ss.job} 신입")
    for msg in ss.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if ss.stage == "done":
        st.success("면접이 끝났습니다. 평가는 다음 단계에서 이어집니다.")
        return

    answer = st.chat_input("답변을 입력하세요 ('종료' 입력 시 면접 종료)")
    if answer is None:
        return
    if not answer.strip():
        st.info("답변을 입력해 주세요.")
        return
    with st.chat_message("user"):
        st.markdown(answer)
    if handle_answer(answer):
        st.rerun()


if ss.stage == "start":
    start_screen()
else:
    interview_screen()
