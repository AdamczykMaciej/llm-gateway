"""Eval toolkit: schema, checks, blind packets, aggregation, calibration, log, runner."""

import json
from pathlib import Path

import pytest

from llm_gateway.evals import (
    Case,
    CheckResult,
    HumanLabel,
    Judgment,
    RunLog,
    Sample,
    Turn,
    aggregate,
    build_packets,
    check_case,
    cohen_kappa,
    collect_judgments,
    compare,
    default_rubric,
    dump_cases,
    dump_samples,
    get_check,
    judge_vs_human,
    load_cases,
    load_labels,
    load_samples,
    register_check,
    registered_checks,
    run_cases,
    save_labels,
    set_language_detector,
    verdicts_from_judgments,
)
from llm_gateway.evals import agreement as agree
from llm_gateway.evals.stats import by_tag, pooled


def case(cid: str = "c1", checks: tuple[dict, ...] = (), tags: tuple[str, ...] = ("t",), **kw):
    return Case(
        cid,
        (Turn("user", "hi"),),
        must=("be nice",),
        must_not=("rude",),
        tags=tags,
        checks=checks,
        **kw,
    )


def test_case_roundtrip_and_shorthand(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    cs = [case("a"), Case("b", (Turn("user", "x"), Turn("assistant", "y"), Turn("user", "z")))]
    dump_cases(p, cs)
    back = load_cases(p)
    assert back == cs and back[1].is_multi_turn and not back[0].is_multi_turn
    short = Case.from_dict({"id": "s", "message": "hello", "lang": "pl"})
    assert short.turns == (Turn("user", "hello"),) and short.language == "pl"


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"id": "x"},
        {"id": "x", "turns": [{"role": "bot", "content": "a"}]},
        {"id": "x", "turns": [{"role": "assistant", "content": "a"}]},
    ],
)
def test_case_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        Case.from_dict(bad)


def test_load_errors(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text('{"id": "a", "message": "m"}\n\n{"id": "a", "message": "m"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(p)
    p.write_text("{nope\n")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_cases(p)


def test_samples_roundtrip(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    ss = [Sample("a", "v1", ("r1", "r2"), 1)]
    dump_samples(p, ss)
    assert load_samples(p) == ss
    assert Sample.from_dict({"case_id": "a", "variant": "v", "reply": "x"}).replies == ("x",)


def test_generic_checks():
    c = case(
        checks=(
            {"name": "max_chars", "limit": 5},
            {"name": "max_words", "limit": 1},
            {"name": "forbidden_patterns", "patterns": [r"\bgreat\b"]},
            {"name": "required_patterns", "patterns": ["ok"], "scope": "all"},
        )
    )
    res = {r.name: r for r in check_case(c, ["Great answer", "ok"])}
    assert not any(r.passed for r in res.values())
    good = check_case(c, ["ok", "ok"])
    assert all(r.passed for r in good)
    last = check_case(
        case(), ["a", "ok"], [{"name": "required_patterns", "patterns": ["ok"], "scope": "last"}]
    )
    assert last[0].passed
    anyscope = check_case(case(), ["a"], [{"name": "required_patterns", "patterns": ["zz"]}])
    assert not anyscope[0].passed
    cs = check_case(
        case(),
        ["Foo"],
        [{"name": "forbidden_patterns", "patterns": ["foo"], "case_sensitive": True}],
    )
    assert cs[0].passed
    assert check_case(case(), [], [{"name": "max_chars", "limit": 1}])[0].passed


def test_repetition_check():
    c = case(checks=({"name": "repetition"},))
    same = "You should update your CV today. Then apply to five roles."
    assert not check_case(c, [same, same])[0].passed
    assert check_case(c, [same, "Totally different words appear in this second reply now."])[
        0
    ].passed
    assert check_case(c, ["one"])[0].passed
    r = check_case(c, ["a b", "c d"])  # sentences too short to compare
    assert r[0].passed


def test_language_check_hook():
    c = case(checks=({"name": "language"},), language="pl")
    assert "skipped" in check_case(c, ["x"])[0].detail
    set_language_detector(lambda t: "en" if "the" in t else None)
    try:
        assert not check_case(c, ["the cat"])[0].passed
        assert check_case(c, ["kot"])[0].passed
        assert check_case(c, ["the"], [{"name": "language", "expected": "en"}])[1].passed
    finally:
        set_language_detector(None)


def test_custom_check_and_failures():
    @register_check("no_emoji")
    def no_emoji(case, replies, params):
        bad = any(ord(ch) > 0x1F000 for r in replies for ch in r)
        return [CheckResult("no_emoji", not bad, "emoji" if bad else "")]

    assert "no_emoji" in registered_checks() and get_check("no_emoji") is no_emoji
    c = case(checks=({"name": "no_emoji"}, {"name": "missing_check"}, {"name": "max_chars"}))
    res = check_case(c, ["hi \U0001f600"])
    by = {r.name: r for r in res}
    assert not by["no_emoji"].passed
    assert "unknown check" in by["missing_check"].detail
    assert "check error" in by["max_chars"].detail  # missing 'limit' param is reported


def _packets(tmp_path: Path, seed: int = 1):
    cases = [case("a"), Case("b", (Turn("user", "q1"), Turn("user", "q2")), tags=("m",))]
    samples = [
        Sample(c.id, v, ("r1",) if c.id == "a" else ("r1", "r2"))
        for c in cases
        for v in ("v1", "v2")
    ]
    return cases, samples, build_packets(cases, samples, tmp_path, batch_size=3, seed=seed)


def test_packets_are_blind_and_keyed(tmp_path: Path):
    cases, samples, ps = _packets(tmp_path)
    key = json.loads(ps.key_path.read_text())
    assert len(key) == len(samples) * 2 and ps.key == key
    seen = set()
    for p in ps.packet_paths:
        for it in json.loads(p.read_text()):
            seen.add(it["id"])
            assert "v1" not in it["prompt"] and "v2" not in it["prompt"]
            assert "MUST NOT" in it["prompt"]
    assert seen == set(key)
    assert (tmp_path / "system.txt").read_text()
    two_turn = next(
        i for p in ps.packet_paths for i in json.loads(p.read_text()) if "q2" in i["prompt"]
    )
    assert two_turn["prompt"].count("REPLY UNDER EVALUATION") == 2
    # same seed -> same ids
    assert _packets(tmp_path / "again")[2].key.keys() == ps.key.keys()


def test_packet_validation(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown cases"):
        build_packets([case()], [Sample("zz", "v", ("r",))], tmp_path)
    with pytest.raises(ValueError, match=">= 1"):
        build_packets([case()], [], tmp_path, batch_size=0)


def _write_outputs(ps, score_for, wrap=False):
    for p in ps.packet_paths:
        items = json.loads(p.read_text())
        rows = []
        for it in items:
            k = ps.key[it["id"]]
            rows.append(
                {"id": it["id"], "helpfulness": score_for(k), "must_not_hit": False, "note": "n"}
            )
        text = json.dumps(rows)
        p.with_name("out_" + p.name).write_text(f"```json\n{text}\n```" if wrap else text)


def test_collect_and_aggregate_ab(tmp_path: Path):
    cases, _, ps = _packets(tmp_path)
    _write_outputs(ps, lambda k: 4 if k["variant"] == "v1" else 2, wrap=True)
    js = collect_judgments(tmp_path)
    assert len(js) == 8
    stats = aggregate(
        js, {("a", "v1"): True, ("a", "v2"): False, ("b", "v1"): True, ("b", "v2"): True}
    )
    cmp = compare(stats, "v1", "v2", cases)
    assert cmp.pooled_delta["helpfulness"] == -2
    assert sorted(cmp.worse) == ["a", "b"] and cmp.better == []
    assert cmp.mech_regressions == ["a"]
    assert cmp.tag_delta["t"]["helpfulness"] == -2
    assert pooled(stats, "v1") == {"helpfulness": 4}
    assert by_tag(stats, cases, "v2")["m"]["helpfulness"] == 2
    better = compare(stats, "v2", "v1", cases)
    assert sorted(better.better) == ["a", "b"]
    assert aggregate(js)[0].mech_pass is None


def test_collect_detects_gaps_and_unknown(tmp_path: Path):
    _, _, ps = _packets(tmp_path)
    with pytest.raises(ValueError, match="no judgment"):
        collect_judgments(tmp_path)
    first = ps.packet_paths[0]
    first.with_name("out_x.json").write_text(json.dumps([{"id": "nope", "helpfulness": 3}]))
    with pytest.raises(ValueError, match="not in key.json"):
        collect_judgments(tmp_path)
    first.with_name("out_x.json").write_text("{bad")
    with pytest.raises(ValueError, match="invalid judge JSON"):
        collect_judgments(tmp_path)
    first.with_name("out_x.json").write_text(json.dumps({"id": next(iter(ps.key)), "tone": 3}))
    assert len(collect_judgments(tmp_path, require_complete=False)) == 1


def test_kappa_and_agreement():
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
    assert cohen_kappa([True, False], [False, True]) == -1.0
    assert cohen_kappa([True, True], [True, True]) is None
    assert cohen_kappa([], []) is None and agree([], []) is None
    assert agree([1, 2], [1, 3]) == 0.5
    with pytest.raises(ValueError):
        cohen_kappa([1], [])
    with pytest.raises(ValueError):
        agree([1], [])


def test_calibration_flow(tmp_path: Path):
    labels = [
        HumanLabel("1", "a", "v1", ("r",), {"kind": True, "tips": None}),
        HumanLabel("2", "b", "v1", ("r",), {"kind": False}),
        HumanLabel("3", "zz", "v1", ("r",), {"kind": True}),  # unjudged: skipped
    ]
    p = tmp_path / "labels.json"
    save_labels(p, labels)
    assert load_labels(p) == labels
    js = [
        Judgment("a", "v1", 0, 1, {"tone": 5}),
        Judgment("a", "v1", 0, 2, {"tone": 4}),
        Judgment("b", "v1", 0, 1, {"tone": 2, "other": 1}),
    ]
    v = verdicts_from_judgments(js, {"tone": 4, "absent": 3})
    assert v[("a", "v1")] == {"tone": True} and v[("b", "v1")] == {"tone": False}
    out = judge_vs_human(labels, {k: {"kind": x["tone"]} for k, x in v.items()})
    assert out["kind"]["n"] == 2 and out["kind"]["agreement"] == 1.0 and out["kind"]["kappa"] == 1.0


def test_runlog(tmp_path: Path):
    log = RunLog(tmp_path / "sub" / "log.md")
    assert log.entries() == []
    log.append("v2 vs v1", "fine", {"n": 3})
    log.append("again", "ok")
    assert "v2 vs v1" in (tmp_path / "sub" / "log.md").read_text()
    es = log.entries()
    assert [e["title"] for e in es] == ["v2 vs v1", "again"] and es[0]["data"] == {"n": 3}


def test_run_cases_multi_turn():
    calls: list[list[dict[str, str]]] = []

    def fn(messages):
        calls.append(messages)
        return f"reply{len(calls)}"

    cs = [
        case("single"),
        Case(
            "multi",
            (
                Turn("user", "a"),
                Turn("user", "b"),
                Turn("assistant", "scripted"),
                Turn("user", "c"),
            ),
        ),
    ]
    out = run_cases(cs, fn, "v1", samples=2)
    assert [s.replies for s in out if s.case_id == "single"] == [("reply1",), ("reply2",)]
    multi = [s for s in out if s.case_id == "multi"]
    assert len(multi) == 2 and len(multi[0].replies) == 2  # b is followed by a scripted turn
    assert calls[3][-5:] == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "reply3"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "scripted"},
        {"role": "user", "content": "c"},
    ]
    assert default_rubric().dimensions


def test_compare_ignores_noise_and_missing_dimensions():
    from llm_gateway.evals.stats import CaseStats

    stats = [
        CaseStats("x", "a", 1, {"d": 3.0}, None, False),
        CaseStats("x", "b", 1, {"d": 3.1}, None, False),
        CaseStats("y", "a", 1, {"d": 3.0}, None, False),
        CaseStats("y", "b", 1, {"e": 1.0}, None, False),
    ]
    cmp = compare(stats, "a", "b")
    assert cmp.worse == [] and cmp.better == [] and cmp.per_case_delta["y"] == {}


def test_prompt_with_scripted_assistant_turn():
    from llm_gateway.evals.judge import render_prompt

    c = Case("m", (Turn("user", "a"), Turn("assistant", "scripted"), Turn("user", "b")))
    p = render_prompt(c, Sample("m", "v", ("only",)), default_rubric())
    assert p.count("REPLY UNDER EVALUATION") == 1 and "[ASSISTANT] scripted" in p
