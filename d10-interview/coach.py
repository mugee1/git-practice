"""질문 고르기 · 진행 판단 등 AI 없이 하는 계산. (API 호출 없음)"""

import json
import random
from pathlib import Path

import pandas as pd

from prompts import CONTEXT, EVALUATOR, INTERVIEWER, PERSONAS, RETRY

TOTAL_QUESTIONS = 5
MAX_FOLLOW_UPS = 1        # 질문 하나당
MAX_TOTAL_FOLLOW_UPS = 3  # 면접 전체
UNDERSTANDING_MIN, UNDERSTANDING_MAX = 0, 100
PANEL_MIN, PANEL_MAX = 0, 10          # 위원 점수
CRITERION_MIN, CRITERION_MAX = 0, 5   # 기준 점수
CRITERIA = {"specific": "구체성", "relevance": "직무 연관", "structure": "구조", "evidence": "근거 · 검증"}
ANSWER_MAX = CRITERION_MAX * len(CRITERIA)  # 20
REWRITE_THRESHOLD = 14                # 답변 점수가 이보다 낮을 때만 고친 답을 보여 줍니다

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


def _fill(text: str, job: str, company: str, experience: str) -> str:
    # 프롬프트 안의 {역량명} 같은 중괄호는 그대로 두려고 format 대신 replace 를 씁니다
    for key, value in {"company": company, "job": job, "experience": experience}.items():
        text = text.replace("{" + key + "}", value)
    return text


def build_instructions(job: str, company: str, experience: str) -> str:
    """페르소나 + INTERVIEWER + CONTEXT 를 붙이고 {company} 등을 채웁니다."""
    return _fill("\n\n".join([PERSONAS[job], INTERVIEWER, CONTEXT]), job, company, experience)


def build_eval_instructions(job: str, company: str, experience: str) -> str:
    """EVALUATOR 의 {context} 자리에 CONTEXT 를 넣고 채웁니다."""
    return _fill(EVALUATOR.replace("{context}", CONTEXT), job, company, experience)


def build_turn_input(questions: list[dict], index: int, follow_ups_used: int,
                     total_follow_ups: int, messages: list[dict]) -> str:
    """면접관에게 넘길 이번 차례의 상태(데이터)를 JSON 문자열로 만듭니다."""
    next_question = questions[index + 1]["질문"] if index + 1 < len(questions) else None
    return json.dumps({
        "현재 메인 질문 번호": f"{index + 1}/{TOTAL_QUESTIONS}",
        "현재 메인 질문": questions[index]["질문"],
        "보는 점": questions[index]["보는 점"],
        "이 질문에서 한 꼬리질문 수": follow_ups_used,
        "면접 전체에서 남은 꼬리질문 수": max(0, MAX_TOTAL_FOLLOW_UPS - total_follow_ups),
        "꼬리질문 가능": can_follow_up(True, follow_ups_used, total_follow_ups),
        "다음 메인 질문": next_question,
        "대화 기록": messages,
    }, ensure_ascii=False)


def can_follow_up(ai_wants_follow_up: bool, follow_ups_used: int, total_follow_ups: int) -> bool:
    """AI 가 더 묻자고 해도 질문당 MAX_FOLLOW_UPS, 전체 MAX_TOTAL_FOLLOW_UPS 를 넘으면 코드가 막습니다."""
    return (bool(ai_wants_follow_up)
            and follow_ups_used < MAX_FOLLOW_UPS
            and total_follow_ups < MAX_TOTAL_FOLLOW_UPS)


def clamp(value, low: int, high: int) -> int:
    """범위 밖 점수는 잘라 씁니다. 숫자가 아니면 최솟값."""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return low
    return max(low, min(high, value))


def clamp_understanding(value) -> int:
    return clamp(value, UNDERSTANDING_MIN, UNDERSTANDING_MAX)


def main_question_text(index: int, questions: list[dict]) -> str:
    return f"[질문 {index + 1}/{TOTAL_QUESTIONS}] {questions[index]['질문']}"


# ---------- 평가 ----------
def answered_numbers(messages: list[dict]) -> list[int]:
    """지원자가 실제로 답한 메인 질문 번호들."""
    return sorted({m["no"] for m in messages if m["role"] == "user" and "no" in m})


def build_eval_input(questions: list[dict], messages: list[dict]) -> str:
    """답한 질문마다 질문 · 보는 점 · 대화를 묶어 JSON 문자열로 만듭니다."""
    records = []
    for no in answered_numbers(messages):
        records.append({
            "no": no,
            "메인 질문": questions[no - 1]["질문"],
            "보는 점": questions[no - 1]["보는 점"],
            "대화": [
                {"화자": "지원자" if m["role"] == "user" else "면접관", "내용": m["content"]}
                for m in messages if m.get("no") == no
            ],
        })
    return json.dumps({"면접 기록": records}, ensure_ascii=False)


def panel_average(panel: list[dict]) -> float:
    """위원 점수 평균 (0~10). AI 가 쓴 합계 · 평균은 쓰지 않습니다."""
    scores = [clamp(p.get("score"), PANEL_MIN, PANEL_MAX) for p in panel]
    return round(sum(scores) / len(scores), 1) if scores else 0.0


def answer_total(answer: dict) -> int:
    """기준 4개 합 (0~20)."""
    return sum(clamp(answer.get(key), CRITERION_MIN, CRITERION_MAX) for key in CRITERIA)


def needs_rewrite(total: int) -> bool:
    return total < REWRITE_THRESHOLD


def tidy_evaluation(result: dict, answered: list[int]) -> dict:
    """점수를 잘라 쓰고, 합계 · 평균을 코드로 다시 계산해 붙입니다."""
    panel = [{**p, "score": clamp(p.get("score"), PANEL_MIN, PANEL_MAX)}
             for p in result.get("panel", [])]
    answers = []
    for a in result.get("answers", []):
        if a.get("no") not in answered:  # 답하지 않은 번호는 버립니다
            continue
        scores = {key: clamp(a.get(key), CRITERION_MIN, CRITERION_MAX) for key in CRITERIA}
        total = answer_total(scores)
        answers.append({**a, **scores, "total": total, "show_rewrite": needs_rewrite(total)})
    answers.sort(key=lambda a: a["no"])
    return {
        "panel": panel,
        "panel_average": panel_average(panel),
        "conflict": result.get("conflict", ""),
        "agree": result.get("agree", ""),
        "answers": answers,
        "next": result.get("next", []),
    }


# ---------- 재도전 ----------
def answer_text(messages: list[dict], no: int) -> str:
    """그 질문에 지원자가 한 답(꼬리 답 포함)을 이어 붙입니다."""
    return "\n\n".join(m["content"] for m in messages if m["role"] == "user" and m.get("no") == no)


def retry_order(answers: list[dict]) -> list[dict]:
    """점수 낮은 순 (같으면 질문 번호 순)."""
    return sorted(answers, key=lambda a: (a["total"], a["no"]))


def build_retry_instructions(job: str, company: str, experience: str) -> str:
    """RETRY 의 {context} 자리에 CONTEXT 를 넣고 채웁니다."""
    return _fill(RETRY.replace("{context}", CONTEXT), job, company, experience)


def build_retry_input(question: dict, answer: str) -> str:
    return json.dumps({
        "질문": question["질문"],
        "보는 점": question["보는 점"],
        "지원자의 답": answer,
    }, ensure_ascii=False)


def tidy_retry(result: dict) -> dict:
    """점수를 잘라 쓰고 합계를 코드로 계산합니다."""
    scores = {key: clamp(result.get(key), CRITERION_MIN, CRITERION_MAX) for key in CRITERIA}
    return {**scores, "total": answer_total(scores), "weak": result.get("weak", "")}


def compare(before: dict, after: dict) -> dict:
    """전 → 후 합계 차이와 기준별 전/후 표를 만듭니다."""
    rows = [
        {"기준": label, "전": before[key], "후": after[key], "차이": after[key] - before[key]}
        for key, label in CRITERIA.items()
    ]
    return {
        "before_total": before["total"],
        "after_total": after["total"],
        "delta": after["total"] - before["total"],
        "rows": rows,
    }


# ---------- 리포트 (BAB: Before → After → Bridge) ----------
def answers_average(answers: list[dict]) -> float:
    """답변 점수 평균 (0~20)."""
    return round(sum(a["total"] for a in answers) / len(answers), 1) if answers else 0.0


def _cell(text) -> str:
    """마크다운 표 칸이 깨지지 않게 | 와 줄바꿈을 정리합니다."""
    return str(text).replace("|", "/").replace("\n", " ")


def build_report(profile: dict, questions: list[dict], evaluation: dict, retries: dict) -> str:
    """숫자는 모두 코드가 계산한 값만 씁니다. AI 가 쓴 합계는 쓰지 않습니다."""
    answers = evaluation["answers"]
    by_no = {a["no"]: a for a in answers}
    compared = {no: compare(by_no[no], r["score"]) for no, r in sorted(retries.items()) if no in by_no}

    lines = [
        f"# 면접 리포트 — {profile['name']}",
        "",
        f"- 회사 · 직군: {profile['company']} · {profile['job']} 신입",
        f"- 위원 평균: **{evaluation['panel_average']} / {PANEL_MAX}**",
        f"- 답변 점수 평균: **{answers_average(answers)} / {ANSWER_MAX}** (답한 질문 {len(answers)}개)",
    ]
    if compared:
        total_delta = sum(c["delta"] for c in compared.values())
        lines.append(f"- 재도전: {len(compared)}개 질문, 합계 차이 **{total_delta:+d}점**")

    lines += ["", "## Before — 면접 답변 점수", "",
              "| 질문 | " + " | ".join(CRITERIA.values()) + f" | 합계 (/{ANSWER_MAX}) | 가장 약한 점 |",
              "| --- " * (len(CRITERIA) + 3) + "|"]
    for a in answers:
        scores = " | ".join(str(a[key]) for key in CRITERIA)
        lines.append(f"| {a['no']}. {_cell(questions[a['no'] - 1]['질문'])} | {scores} | {a['total']} | {_cell(a['weak'])} |")

    lines += ["", "## After — 재도전 전후", ""]
    if compared:
        lines += [f"| 질문 | 전 (/{ANSWER_MAX}) | 후 (/{ANSWER_MAX}) | 차이 |", "| --- | --- | --- | --- |"]
        for no, c in compared.items():
            lines.append(f"| {no}. {_cell(questions[no - 1]['질문'])} | {c['before_total']} | {c['after_total']} | {c['delta']:+d} |")
        for no, c in compared.items():
            lines += ["", f"**질문 {no} 기준별**", "", "| 기준 | 전 | 후 | 차이 |", "| --- | --- | --- | --- |"]
            lines += [f"| {r['기준']} | {r['전']} | {r['후']} | {r['차이']:+d} |" for r in c["rows"]]
    else:
        lines.append("재도전 기록이 없습니다.")

    lines += ["", "## Bridge — 다음 연습에서 고칠 것", ""]
    lines += [f"- {item}" for item in evaluation["next"]]
    if answers:
        weakest = retry_order(answers)[0]
        lines.append(f"- 가장 낮은 답변: 질문 {weakest['no']} ({weakest['total']} / {ANSWER_MAX}) — 이 질문부터 다시 연습")
    return "\n".join(lines) + "\n"
