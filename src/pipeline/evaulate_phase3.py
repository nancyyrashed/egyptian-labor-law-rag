"""
Evaluation.

Runs a fixed Q&A test set through the RAG pipeline (load_resources / ask,
imported directly from ask.py - no logic duplicated) and reports, per
question and in aggregate:

  - retrieval hit rate      (was the expected article among the
                              top-k chunks Chroma actually returned?)
  - citation accuracy       (did the model cite the expected
                              article(s), out of what it cited?)
  - hallucination rate      (did the model cite an article number
                              that was never even retrieved? - this
                              reuses ask.py's own hallucination check,
                              it isn't recomputed here)
  - abstention correctness  (did it abstain exactly when it should:
                              yes for out-of-scope questions, no for
                              in-scope ones?)

Ground truth note:
Every expected_articles value below was checked directly against
labor_law_articles.json, not filled in from general legal knowledge. A
few things worth knowing:

  - "Work-injury compensation" and "iddah/widowhood leave" aren't in
    this test set because this law doesn't answer them itself -
    Article 1's definitions clause explicitly defers "إصابة العمل"
    to the Social Insurance Law (148/2019) instead of defining it
    here, and no iddah-leave provision exists in the corpus at all.
    Asking the system these would really test "does it abstain on
    something adjacent but not covered," which would be a good
    addition but needs a clear scoring rule (citing Article 1's
    deferral is arguably correct, not a hallucination). They were
    left out rather than scored against guessed ground truth.
  - "Trade union multi-membership" was dropped for the same reason -
    this labor law doesn't legislate it (it's Trade Union Law 213/2017
    territory), so there is no article to check the system against.
  - "Fixed-term contract max duration" and "end-of-service gratuity"
    don't have a single clean answer in this law: it doesn't cap
    fixed-term contract length outright, and general gratuity was
    mostly replaced by the pension system. They were reworded to the
    specific provisions that do exist (Article 154's right to end a
    contract longer than five years, Article 172's gratuity for
    service continued past age 60).

Run (from the project root):
    python src/pipeline/evaulate_phase3.py

Outputs:
    data/processed/phase3_eval_results.json   (full per-question detail)
    data/processed/phase3_eval_summary.md     (results table)
"""

import json
import os
import sys
import time

# Reuse the pipeline in ask.py directly rather than re-implementing
# retrieval/generation/citation logic here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "pipeline"))
from ask import load_resources, ask  # noqa: E402


RESULTS_JSON_PATH = "data/processed/phase3_eval_results.json"
SUMMARY_MD_PATH = "data/processed/phase3_eval_summary.md"


# ---------------------------------------------------------------
# TEST SET - 20 questions: 15 in-scope (expected_articles verified
# against labor_law_articles.json) and 5 out-of-scope (should abstain).
# Topics are spread across the law rather than clustered around the
# four articles used in the smoke tests in ask.py.
# ---------------------------------------------------------------

TEST_SET = [
    # --- in-scope: verified against the corpus ---
    {
        "question": "كم مدة فترة الاختبار المسموح بها للعامل؟",
        "expected_articles": [90],
        "category": "in_scope",
        "topic": "probation period",
    },
    {
        "question": "ما هو الحد الأقصى لساعات العمل اليومية؟",
        "expected_articles": [117],
        "category": "in_scope",
        "topic": "max daily working hours",
    },
    {
        "question": "كم مدة إجازة الوضع للعاملة الحامل؟",
        "expected_articles": [54],
        "category": "in_scope",
        "topic": "maternity leave duration",
    },
    {
        "question": "ما هي مدة الإجازة السنوية للعامل؟",
        "expected_articles": [124],
        "category": "in_scope",
        "topic": "annual leave duration",
    },
    {
        "question": "ما هي مدة مهلة الإخطار بإنهاء عقد العمل غير محدد المدة؟",
        # Article 156: 3 months' written notice for either party.
        "expected_articles": [156],
        "category": "in_scope",
        "topic": "notice period for indefinite contract termination",
    },
    {
        "question": "هل يجوز تشغيل الطفل بين الساعة السابعة مساءً والسابعة صباحًا؟",
        # Article 65: explicit night-work ban for children, 7pm-7am.
        "expected_articles": [65],
        "category": "in_scope",
        "topic": "night work ban for children",
    },
    {
        "question": "ما هو الحد الأدنى لسن تشغيل الأطفال؟",
        # Article 62: minimum working age is 15 (training allowed from 14).
        "expected_articles": [62],
        "category": "in_scope",
        "topic": "minimum working age",
    },
    {
        "question": "ما هو أجر العامل إذا تم تشغيله في يوم الراحة الأسبوعية؟",
        # Article 121: equal day's pay as compensation + a substitute
        # rest day the following week. (Article 120 defines the paid
        # weekly rest itself and may legitimately get cited alongside it.)
        "expected_articles": [121],
        "category": "in_scope",
        "topic": "pay for work on the weekly rest day",
    },
    {
        "question": "ما هي حالات الخطأ الجسيم التي يجوز معها فصل العامل؟",
        # Article 148: the exhaustive list of gross-misconduct grounds.
        "expected_articles": [148],
        "category": "in_scope",
        "topic": "gross-misconduct grounds for dismissal",
    },
    {
        "question": "ما هي العقوبات التأديبية التي يجوز لصاحب العمل توقيعها على العامل؟",
        # Article 139: the 8-item disciplinary sanctions list.
        "expected_articles": [139],
        "category": "in_scope",
        "topic": "disciplinary sanctions",
    },
    {
        "question": "ما هي مدة الإجازة المرضية للعامل وكم يستحق من أجر خلالها؟",
        # Article 131: sick leave, pay per Social Insurance Law rates,
        # plus the industrial-establishments special case (3mo full +
        # 6mo at 85%).
        "expected_articles": [131],
        "category": "in_scope",
        "topic": "sick leave duration and pay",
    },
    {
        "question": "ماذا يحدث إذا تجاوزت مدة عقد العمل محدد المدة خمس سنوات؟",
        # Article 154: right to end without compensation after 5 years,
        # with 3 months' notice.
        "expected_articles": [154],
        "category": "in_scope",
        "topic": "fixed-term contract exceeding five years",
    },
    {
        "question": "هل يستحق العامل مكافأة عن مدة عمله بعد سن الستين؟",
        # Article 172: gratuity for service continued past age 60.
        "expected_articles": [172],
        "category": "in_scope",
        "topic": "gratuity for work after age 60",
    },
    {
        "question": "كم ساعة راحة أسبوعية يستحقها العامل؟",
        # Article 120: 24 consecutive hours after at most 6 working days,
        # paid.
        "expected_articles": [120],
        "category": "in_scope",
        "topic": "weekly rest entitlement",
    },
    {
        "question": "هل تستحق العاملة فترات راحة للرضاعة بعد إجازة الوضع؟",
        # Article 56: two extra nursing breaks (>= 30 min each) for two
        # years after birth, counted as working hours, no pay cut.
        "expected_articles": [56],
        "category": "in_scope",
        "topic": "breastfeeding breaks",
    },

    # --- out-of-scope: should abstain, not answer from general knowledge ---
    {
        "question": "ما هي عقوبة السرقة في قانون العقوبات؟",
        "expected_articles": [],
        "category": "out_of_scope",
        "topic": "criminal theft penalty",
    },
    {
        "question": "كيف يتم احتساب ضريبة الدخل على الأفراد؟",
        "expected_articles": [],
        "category": "out_of_scope",
        "topic": "personal income tax calculation",
    },
    {
        "question": "ما هي إجراءات تسجيل عقد إيجار سكني؟",
        "expected_articles": [],
        "category": "out_of_scope",
        "topic": "residential lease registration",
    },
    {
        "question": "ما هي شروط الطلاق للضرر في القانون المصري؟",
        "expected_articles": [],
        "category": "out_of_scope",
        "topic": "divorce on grounds of harm",
    },
    {
        "question": "ما هي غرامة تجاوز السرعة المقررة في قانون المرور؟",
        "expected_articles": [],
        "category": "out_of_scope",
        "topic": "traffic speeding fine",
    },
]


def check_test_set_ready(test_set):
    """
    Refuse to silently evaluate against placeholder ground truth, in
    case this test set is extended later and a new entry is added
    without filling in a real, corpus-verified article number.
    """
    unverified = [
        t["topic"] for t in test_set
        if t["category"] == "in_scope" and None in t["expected_articles"]
    ]
    if unverified:
        print("The following in-scope questions still have UNVERIFIED")
        print("expected_articles (placeholder None) — fill these in by")
        print("checking labor_law_articles.json before running a real eval:")
        for topic in unverified:
            print(f"  - {topic}")
        print()
        answer = input("Continue anyway with placeholders skipped in scoring? [y/N] ")
        if answer.strip().lower() != "y":
            sys.exit(1)


def evaluate_question(embed_model, collection, groq_client, case):
    result = ask(
        embed_model, collection, groq_client, case["question"], verbose=False
    )

    expected = set(a for a in case["expected_articles"] if a is not None)
    has_ground_truth = bool(expected) or case["category"] == "out_of_scope"

    retrieved = set(result["retrieved_articles"])
    cited = set(result["cited_articles"])

    retrieval_hit = None
    citation_correct = None
    abstention_correct = None

    if case["category"] == "out_of_scope":
        # Correct behavior: abstain, cite nothing.
        abstention_correct = result["is_abstention"] and not cited
    elif expected:
        retrieval_hit = bool(expected & retrieved)
        # Citation is "correct" if every expected article that was
        # actually retrievable got cited. (Can't cite an article
        # that retrieval never surfaced — that's a retrieval failure,
        # not a citation failure, so it's scored separately above.)
        expected_and_retrieved = expected & retrieved
        citation_correct = (
            expected_and_retrieved.issubset(cited) if expected_and_retrieved else None
        )
        abstention_correct = not result["is_abstention"]

    return {
        "question": case["question"],
        "topic": case["topic"],
        "category": case["category"],
        "expected_articles": sorted(expected) if expected else [],
        "retrieved_articles": result["retrieved_articles"],
        "cited_articles": result["cited_articles"],
        "hallucinated_citations": result["hallucinated_citations"],
        "is_abstention": result["is_abstention"],
        "retrieval_hit": retrieval_hit,
        "citation_correct": citation_correct,
        "abstention_correct": abstention_correct,
        "has_ground_truth": has_ground_truth,
        "answer": result["answer"],
        # Fill this in yourself after reading the answer text — did it
        # actually address the question well, not just cite the right
        # article? Kept separate because no script here can judge that
        # honestly; this project's whole ethos has been "verify against
        # real output," not "trust an automated proxy for correctness."
        "manual_correctness_notes": "",
    }


def print_summary_table(results):
    in_scope = [r for r in results if r["category"] == "in_scope"]
    out_scope = [r for r in results if r["category"] == "out_of_scope"]

    scored_retrieval = [r for r in in_scope if r["retrieval_hit"] is not None]
    scored_citation = [r for r in in_scope if r["citation_correct"] is not None]
    scored_abstention_in = [r for r in in_scope if r["abstention_correct"] is not None]

    retrieval_hit_rate = (
        sum(r["retrieval_hit"] for r in scored_retrieval) / len(scored_retrieval)
        if scored_retrieval else None
    )
    citation_accuracy = (
        sum(r["citation_correct"] for r in scored_citation) / len(scored_citation)
        if scored_citation else None
    )
    false_answer_rate = (
        1 - sum(r["abstention_correct"] for r in scored_abstention_in) / len(scored_abstention_in)
        if scored_abstention_in else None
    )
    correct_abstention_rate = (
        sum(r["abstention_correct"] for r in out_scope) / len(out_scope)
        if out_scope else None
    )
    total_hallucinations = sum(len(r["hallucinated_citations"]) for r in results)

    print("\n" + "=" * 70)
    print("PHASE 3 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total questions: {len(results)} "
          f"({len(in_scope)} in-scope, {len(out_scope)} out-of-scope)")
    if retrieval_hit_rate is not None:
        print(f"Retrieval top-5 hit rate:        {retrieval_hit_rate:.0%} "
              f"({sum(r['retrieval_hit'] for r in scored_retrieval)}/{len(scored_retrieval)})")
    if citation_accuracy is not None:
        print(f"Citation accuracy (when hit):     {citation_accuracy:.0%} "
              f"({sum(r['citation_correct'] for r in scored_citation)}/{len(scored_citation)})")
    if false_answer_rate is not None:
        print(f"False-abstention rate (in-scope): {false_answer_rate:.0%} "
              "(should be 0% — model wrongly refused to answer)")
    if correct_abstention_rate is not None:
        print(f"Correct abstention (out-of-scope): {correct_abstention_rate:.0%} "
              f"({sum(r['abstention_correct'] for r in out_scope)}/{len(out_scope)})")
    print(f"Total hallucinated citations across all runs: {total_hallucinations} "
          "(should be 0)")

    print("\nPer-question detail:")
    print(f"{'Topic':<38} {'Cat':<7} {'RetHit':<7} {'CiteOK':<7} {'Abstain'}")
    print("-" * 75)
    for r in results:
        ret = "-" if r["retrieval_hit"] is None else ("YES" if r["retrieval_hit"] else "NO")
        cite = "-" if r["citation_correct"] is None else ("YES" if r["citation_correct"] else "NO")
        abst = "-" if r["abstention_correct"] is None else ("YES" if r["abstention_correct"] else "NO")
        print(f"{r['topic']:<38} {r['category']:<7} {ret:<7} {cite:<7} {abst}")


def write_summary_markdown(results, path):
    lines = [
        "# Phase 3 — Evaluation Results\n",
        f"Total questions: {len(results)}\n",
        "| Topic | Category | Retrieval hit | Citation correct | Abstention correct | Hallucinated |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        def fmt(v):
            return "-" if v is None else ("Yes" if v else "No")
        halluc = ", ".join(str(a) for a in r["hallucinated_citations"]) or "-"
        lines.append(
            f"| {r['topic']} | {r['category']} | {fmt(r['retrieval_hit'])} | "
            f"{fmt(r['citation_correct'])} | {fmt(r['abstention_correct'])} | {halluc} |"
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nMarkdown summary table saved to {path}")


def main():
    check_test_set_ready(TEST_SET)

    print("Loading model, Chroma collection, and Groq client...")
    embed_model, collection, groq_client = load_resources()

    results = []
    for i, case in enumerate(TEST_SET, start=1):
        print(f"[{i}/{len(TEST_SET)}] {case['topic']}...")
        try:
            result = evaluate_question(embed_model, collection, groq_client, case)
        except Exception as e:
            # Same resilience pattern as ask.py's main(): one failure
            # shouldn't cost every result computed before it, and this
            # loop makes a full pass of Groq calls back to back.
            print(f"  [ERROR] {type(e).__name__}: {e}")
            result = {
                "question": case["question"],
                "topic": case["topic"],
                "category": case["category"],
                "error": str(e),
            }
        results.append(result)

        os.makedirs(os.path.dirname(RESULTS_JSON_PATH), exist_ok=True)
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Small pause between questions. Not a fix for the underlying
        # rate limit (Groq's free-tier TPM cap), just spacing requests
        # out so a batch of 20 back-to-back calls - now carrying more
        # tokens per call since top_k went 5->8 - doesn't burst past
        # it the way question 6 did in the previous run.
        time.sleep(3)

    ok_results = [r for r in results if "error" not in r]
    print_summary_table(ok_results)
    write_summary_markdown(ok_results, SUMMARY_MD_PATH)

    n_errors = sum(1 for r in results if "error" in r)
    print(f"\nSaved {len(results)} results to {RESULTS_JSON_PATH}"
          f"{f' ({n_errors} failed — see error field)' if n_errors else ''}")


if __name__ == "__main__":
    main()
