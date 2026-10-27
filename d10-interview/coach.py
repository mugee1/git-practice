"""질문 고르기 · 진행 판단 등 AI 없이 하는 계산. (API 호출 없음)"""

import json
import random
from pathlib import Path

import pandas as pd

from prompts import CONTEXT, INTERVIEWER, PERSONAS

TOTAL_QUESTIONS = 5
MAX_FOLLOW_UPS = 1
UNDERSTANDING_MIN, UNDERSTANDING_MAX = 0, 100

QUESTIONS_PATH = Path(__file__).parent / "data" / "questions.csv"
JOBS = list(PERSONAS)
JOB_TYPES = ("직무", "경험", "상황")


def load_questions(path: Path = QUESTIONS_PATH) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _row(row: pd.Series) -> dict:
    return {"유형": row["유형"], "질문": row["질문"], "보는 점": row["보는 점"]}


def pick_questions(df: pd.DataFrame, job: str, seed: int | None = None) -> list[dict]:
    """자기소개 1 + 공통 1 + 직무 1 + 경험 1 + 상황 1 = 5개를 이 순서로 고릅니다."""
    rng = random.Random(seed)
    common = df[df["직군"] == "공통"]
    is_intro = common["질문"].str.contains("자기소개")

    picked = [_row(common[is_intro].iloc[0])]
    picked.append(_row(common[~is_intro].iloc[rng.randrange(len(common[~is_intro]))]))
    job_df = df[df["직군"] == job]
    for kind in JOB_TYPES:
        pool = job_df[job_df["유형"] == kind]
        if pool.empty:
            raise ValueError(f"질문 은행에 '{job} · {kind}' 질문이 없습니다.")
        picked.append(_row(pool.iloc[rng.randrange(len(pool))]))

    assert len(picked) == TOTAL_QUESTIONS
    return picked


def build_instructions(job: str, company: str, experience: str) -> str:
    """페르소나 + INTERVIEWER + CONTEXT 를 붙이고 {company} 등을 채웁니다."""
    text = "\n\n".join([PERSONAS[job], INTERVIEWER, CONTEXT])
    # 프롬프트 안의 {역량명} 같은 중괄호는 그대로 두려고 format 대신 replace 를 씁니다
    for key, value in {"company": company, "job": job, "experience": experience}.items():
        text = text.replace("{" + key + "}", value)
    return text


def build_turn_input(questions: list[dict], index: int, follow_ups_used: int,
                     messages: list[dict]) -> str:
    """면접관에게 넘길 이번 차례의 상태(데이터)를 JSON 문자열로 만듭니다."""
    next_question = questions[index + 1]["질문"] if index + 1 < len(questions) else None
    return json.dumps({
        "현재 메인 질문 번호": f"{index + 1}/{TOTAL_QUESTIONS}",
        "현재 메인 질문": questions[index]["질문"],
        "보는 점": questions[index]["보는 점"],
        "이 질문에서 한 꼬리질문 수": follow_ups_used,
        "꼬리질문 가능": follow_ups_used < MAX_FOLLOW_UPS,
        "다음 메인 질문": next_question,
        "대화 기록": messages,
    }, ensure_ascii=False)


def can_follow_up(ai_wants_follow_up: bool, follow_ups_used: int) -> bool:
    """AI 가 더 묻자고 해도 질문당 MAX_FOLLOW_UPS 를 넘으면 코드가 막습니다."""
    return bool(ai_wants_follow_up) and follow_ups_used < MAX_FOLLOW_UPS


def clamp_understanding(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return UNDERSTANDING_MIN
    return max(UNDERSTANDING_MIN, min(UNDERSTANDING_MAX, value))


def main_question_text(index: int, questions: list[dict]) -> str:
    return f"[질문 {index + 1}/{TOTAL_QUESTIONS}] {questions[index]['질문']}"
