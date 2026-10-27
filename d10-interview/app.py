"""AI 면접 코치 — Streamlit 화면만 둡니다."""

import pandas as pd
import streamlit as st

import coach
from llm import LLMError, ask_json
from prompts import EVAL_SCHEMA, INTERVIEWER_SCHEMA, RETRY_SCHEMA

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
    if st.button("처음부터 다시", width="stretch"):
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
        evaluation=None,
        retries={},  # 질문 번호 → 다시 쓴 답과 점수
        instructions=coach.build_instructions(job, company.strip(), experience.strip()),
        messages=[{
            "role": "assistant",
            "content": "면접을 시작하겠습니다.\n\n" + coach.main_question_text(0, questions),
            "no": 1,
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

    # "no" = 몇 번 메인 질문에 대한 대화인지 (평가 때 답변별로 묶는 데 씁니다)
    ss.messages.append({"role": "user", "content": answer, "no": ss.q_index + 1})
    ss.understanding = coach.clamp_understanding(result.get("understanding"))

    if coach.can_follow_up(result.get("follow_up"), ss.follow_ups_used):
        ss.follow_ups_used += 1
        ss.messages.append({"role": "assistant", "content": result.get("reply", ""), "no": ss.q_index + 1})
        return True

    # 꼬리 질문이 없거나 이미 1번 했으면 코드가 다음 메인 질문으로 넘깁니다
    ss.q_index += 1
    ss.follow_ups_used = 0
    if ss.q_index >= coach.TOTAL_QUESTIONS:
        finish("메인 질문 5개를 모두 마쳤습니다. 수고하셨습니다.")
    else:
        ss.messages.append({
            "role": "assistant",
            "content": coach.main_question_text(ss.q_index, ss.questions),
            "no": ss.q_index + 1,
        })
    return True


def show_messages():
    for msg in ss.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---------- 평가 탭 ----------
def run_evaluation():
    answered = coach.answered_numbers(ss.messages)
    if not answered:
        st.info("평가할 답변이 없습니다. '처음부터 다시'로 면접을 새로 시작해 주세요.")
        return
    instructions = coach.build_eval_instructions(ss.job, ss.company, ss.experience)
    eval_input = coach.build_eval_input(ss.questions, ss.messages)
    with st.spinner("위원 4명이 평가하고 있어요..."):
        try:
            result = ask_json(instructions, eval_input, EVAL_SCHEMA, "evaluation")
        except LLMError as e:
            st.error(str(e))
            return
    ss.evaluation = coach.tidy_evaluation(result, answered)


def evaluation_tab():
    if st.button("위원 4명에게 평가받기", type="primary"):
        run_evaluation()
    ev = ss.evaluation
    if ev is None:
        return

    st.metric("위원 평균", f"{ev['panel_average']} / {coach.PANEL_MAX}")
    st.dataframe(
        pd.DataFrame([
            {"위원": p["role"], "뽑지 말아야 할 이유": p["against"], "좋은 점": p["good"], "점수": p["score"]}
            for p in ev["panel"]
        ]),
        hide_index=True,
        width="stretch",
    )
    col1, col2 = st.columns(2)
    col1.markdown(f"**갈린 곳**\n\n{ev['conflict']}")
    col2.markdown(f"**같은 판단**\n\n{ev['agree']}")

    st.subheader("답변별 평가")
    for a in ev["answers"]:
        question = ss.questions[a["no"] - 1]["질문"]
        with st.expander(f"질문 {a['no']} · {a['total']} / {coach.ANSWER_MAX}점 — {question}"):
            st.dataframe(
                pd.DataFrame([{label: a[key] for key, label in coach.CRITERIA.items()}]),
                hide_index=True,
            )
            st.markdown(f"**가장 약한 점** {a['weak']}")
            # 답변 점수가 기준(14점) 미만일 때만 고친 답을 보여 줍니다
            if a["show_rewrite"]:
                st.markdown("**고친 답**")
                st.info(a["rewrite"])
                if a["check"]:
                    st.markdown("**사람이 확인할 것**")
                    for item in a["check"]:
                        st.markdown(f"- {item}")

    st.subheader("다음 연습에서 고칠 것")
    for item in ev["next"]:
        st.markdown(f"- {item}")


# ---------- 재도전 탭 ----------
def run_retry(no: int, answer: str):
    instructions = coach.build_retry_instructions(ss.job, ss.company, ss.experience)
    retry_input = coach.build_retry_input(ss.questions[no - 1], answer)
    with st.spinner("다시 채점하고 있어요..."):
        try:
            result = ask_json(instructions, retry_input, RETRY_SCHEMA, "retry")
        except LLMError as e:
            st.error(str(e))
            return
    ss.retries[no] = {"answer": answer, "score": coach.tidy_retry(result)}


def retry_tab():
    ev = ss.evaluation
    if ev is None or not ev["answers"]:
        st.info("먼저 '📊 평가' 탭에서 평가를 받아 주세요.")
        return

    by_no = {a["no"]: a for a in ev["answers"]}
    no = st.selectbox(
        "다시 답할 질문 (점수 낮은 순)",
        [a["no"] for a in coach.retry_order(ev["answers"])],
        format_func=lambda n: f"질문 {n} · {by_no[n]['total']} / {coach.ANSWER_MAX}점 — {ss.questions[n - 1]['질문']}",
        key="retry_no",
    )
    before = by_no[no]

    st.metric("지난 점수", f"{before['total']} / {coach.ANSWER_MAX}")
    st.markdown(f"**가장 약한 점** {before['weak']}")
    with st.expander("지난 답 보기"):
        st.markdown(coach.answer_text(ss.messages, no))

    answer = st.text_area("다시 답하기", key=f"retry_answer_{no}", height=180)
    if st.button("다시 채점", type="primary"):
        if not answer.strip():
            st.info("다시 쓴 답을 입력해 주세요.")
        else:
            run_retry(no, answer.strip())

    retry = ss.retries.get(no)
    if retry is None:
        return
    result = coach.compare(before, retry["score"])
    st.divider()
    st.metric("다시 답한 점수", f"{result['after_total']} / {coach.ANSWER_MAX}", delta=result["delta"])
    st.dataframe(pd.DataFrame(result["rows"]), hide_index=True)
    st.markdown(f"**가장 약한 점 (다시 답한 뒤)** {retry['score']['weak']}")


# ---------- 리포트 탭 ----------
def report_tab():
    if ss.evaluation is None:
        st.info("먼저 '📊 평가' 탭에서 평가를 받아 주세요.")
        return
    profile = {"name": ss.name, "company": ss.company, "job": ss.job}
    report = coach.build_report(profile, ss.questions, ss.evaluation, ss.retries)
    st.download_button(
        "📄 리포트 내려받기",
        data=report,
        file_name="면접리포트.md",
        mime="text/markdown",
        type="primary",
    )
    with st.expander("미리 보기", expanded=True):
        st.markdown(report)


def interview_screen():
    st.title("🎤 모의면접")
    st.caption(f"{ss.name} · {ss.company} {ss.job} 신입")

    if ss.stage == "done":
        chat_tab, eval_tab, retry_tab_, report_tab_ = st.tabs(["💬 면접", "📊 평가", "🔁 재도전", "📄 리포트"])
        with chat_tab:
            show_messages()
            st.success("면접이 끝났습니다. '📊 평가' 탭에서 평가를 받아 보세요.")
        with eval_tab:
            evaluation_tab()
        with retry_tab_:
            retry_tab()
        with report_tab_:
            report_tab()
        return

    show_messages()
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
