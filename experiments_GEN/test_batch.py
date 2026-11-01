"""Offline tests of the batch workflow with a fake Message Batches client. No network, no cost.

    python experiments_GEN/test_batch.py
"""

from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_runner as B  # noqa: E402


# --- fake SDK -----------------------------------------------------------------------------------
def ns(**kw):
    return types.SimpleNamespace(**kw)


class FakeBatches:
    """Mimics client.messages.batches: create / retrieve / results. `plan` decides each request's
    outcome on each attempt: {custom_id: ["errored:api_error", "succeeded"]}; default succeeded."""

    def __init__(self, reply, plan=None, in_flight_polls=1):
        self.reply, self.plan, self.polls_left = reply, plan or {}, {}
        self.batches, self.created, self.in_flight_polls = {}, [], in_flight_polls
        self.seen = {}

    def create(self, requests):
        bid = f"msgbatch_{len(self.batches)}"
        for r in requests:  # the SDK accepts plain dicts; check the shape we send
            assert set(r) == {"custom_id", "params"} and r["params"]["model"]
        self.batches[bid] = requests
        self.created.append([r["custom_id"] for r in requests])
        self.polls_left[bid] = self.in_flight_polls
        return ns(id=bid, processing_status="in_progress")

    def retrieve(self, bid):
        n = len(self.batches[bid])
        if self.polls_left[bid] > 0:
            self.polls_left[bid] -= 1
            return ns(processing_status="in_progress", request_counts=ns(processing=n, succeeded=0, errored=0,
                                                                         canceled=0, expired=0))
        outs = [self._outcome(r["custom_id"], peek=True) for r in self.batches[bid]]
        c = {k: sum(o.startswith(k) for o in outs) for k in ("succeeded", "errored", "canceled", "expired")}
        return ns(processing_status="ended", request_counts=ns(processing=0, **c))

    def _outcome(self, cid, peek=False):
        k = self.seen.get(cid, 0)
        seq = self.plan.get(cid, ["succeeded"])
        o = seq[min(k, len(seq) - 1)]
        if not peek:
            self.seen[cid] = k + 1
        return o

    def results(self, bid):
        for r in self.batches[bid]:
            o = self._outcome(r["custom_id"])
            if o == "succeeded":
                text = self.reply(r["params"])
                msg = ns(content=[ns(type="thinking", thinking=""), ns(type="text", text=text)],
                         usage=ns(input_tokens=1000, output_tokens=2000), stop_reason="end_turn",
                         model="claude-haiku-5-5")
                yield ns(custom_id=r["custom_id"], result=ns(type="succeeded", message=msg))
            elif o.startswith("errored"):
                yield ns(custom_id=r["custom_id"], result=ns(type="errored", error=ns(type=o.split(":")[1])))
            else:
                yield ns(custom_id=r["custom_id"], result=ns(type=o))


def fake_client(batches):
    return ns(messages=ns(batches=batches))


def reqs(n, prefix="t"):
    return [{"prefix": prefix, "meta": {"i": i},
             "params": {"model": "claude-haiku-5-5", "max_tokens": 8000, "output_config": {"effort": "medium"},
                        "messages": [{"role": "user", "content": f"q{i}"}]}} for i in range(n)]


def fresh(tmp):
    B.LEDGER = Path(tmp) / "ledger.md"
    return B.Job(Path(tmp) / "job")


# --- tests --------------------------------------------------------------------------------------
def test_ids_and_dedup():
    with tempfile.TemporaryDirectory() as d:
        job = fresh(d)
        r = reqs(3) + reqs(1)  # q0 twice: must collapse
        job.create(r, (900, 3900), "t")
        ids = [x["custom_id"] for x in job.requests()]
        assert len(ids) == 3 and len(set(ids)) == 3
        assert all(len(i) <= 64 and all(c.isalnum() or c in "-_" for c in i) for i in ids)
        job.create(reqs(3), (900, 3900), "t")  # same requests again: no-op
        try:
            job.create(reqs(4), (900, 3900), "t")  # different requests: refused
            raise AssertionError("should refuse")
        except SystemExit:
            pass


def test_plan_makes_no_calls():
    with tempfile.TemporaryDirectory() as d:
        job = fresh(d)
        job.create(reqs(5), (900, 3900), "t")
        fb = FakeBatches(lambda p: "{}")
        job.submit(fake_client(fb), cap_usd=1, confirm=False)
        assert fb.created == [] and not B.LEDGER.exists()


def test_cost_estimate_uses_batch_prices():
    with tempfile.TemporaryDirectory() as d:
        job = fresh(d)
        job.create(reqs(100), (900, 3900), "t")
        expected = 100 * (900 * 0.05e-6 + 3900 * 0.25e-6)
        assert abs(job.estimate() - expected) < 1e-9, job.estimate()


def test_full_cycle_retry_only_failures():
    with tempfile.TemporaryDirectory() as d:
        job = fresh(d)
        job.create(reqs(6), (900, 3900), "t")
        ids = [x["custom_id"] for x in job.requests()]
        plan = {ids[0]: ["errored:api_error", "succeeded"], ids[1]: ["expired", "succeeded"],
                ids[2]: ["errored:invalid_request"], ids[3]: ["errored:overloaded_error"] * 5}
        fb = FakeBatches(lambda p: '{"ok": 1}', plan)
        c = fake_client(fb)
        job.submit(c, cap_usd=1, confirm=True)
        assert job.submit(c, cap_usd=1, confirm=True) == []      # in flight: no second submit
        job.collect(c, wait=True, poll_s=0)
        assert job.summary()["results"] == {"errored": 3, "expired": 1, "succeeded": 2}
        # second round: only 0, 1, 3 go again; 2 (invalid_request) never; successes never
        job.submit(c, cap_usd=1, confirm=True)
        assert sorted(fb.created[1]) == sorted([ids[0], ids[1], ids[3]]), fb.created[1]
        job.collect(c, wait=True, poll_s=0)
        res = job.results()
        assert res[ids[0]]["type"] == res[ids[1]]["type"] == "succeeded"
        # third round: 3 has used MAX_ATTEMPTS=2, so nothing is left to send
        assert job.pending() == [] and job.submit(c, cap_usd=1, confirm=True) == []
        # ledger: one line per batch, actual cost from usage x batch prices
        lines = [l for l in B.LEDGER.read_text().splitlines() if l.startswith("| 20")]
        assert len(lines) == 2
        per_ok = 1000 * 0.05e-6 + 2000 * 0.25e-6
        assert abs(B.ledger_total() - 4 * per_ok) < 1e-6, B.ledger_total()


def test_resume_after_interruption():
    with tempfile.TemporaryDirectory() as d:
        job = fresh(d)
        job.create(reqs(4), (900, 3900), "t")
        fb = FakeBatches(lambda p: "{}", in_flight_polls=3)
        c = fake_client(fb)
        job.submit(c, cap_usd=1, confirm=True)
        job.collect(c, wait=False)            # "crash": batch still in progress
        job2 = B.Job(job.dir)                 # new process reads saved state
        assert job2.state()["batches"][0]["status"] == "in_progress"
        assert job2.submit(c, cap_usd=1, confirm=True) == []   # never double-submits
        job2.collect(c, wait=True, poll_s=0)
        job2.collect(c, wait=True, poll_s=0)  # collecting twice adds nothing
        assert len(B.read_jsonl(job2.res_path)) == 4 and len(fb.created) == 1
        assert len([l for l in B.LEDGER.read_text().splitlines() if l.startswith("| 20")]) == 1


def test_spending_guards():
    with tempfile.TemporaryDirectory() as d:
        job = fresh(d)
        job.create(reqs(1000), (900, 3900), "t")   # est ~ $1.02
        fb = FakeBatches(lambda p: "{}")
        for cap in (0.5,):
            try:
                job.submit(fake_client(fb), cap_usd=cap, confirm=True)
                raise AssertionError("cap not enforced")
            except SystemExit:
                pass
        B.LEDGER.write_text(B.HEADER + "| 2026-10-31T00:00:00+00:00 | x | b | 1 / 0 / 0 | 1 | 99.500000 | 0 | 0 |\n")
        try:
            job.submit(fake_client(fb), cap_usd=5, confirm=True)
            raise AssertionError("project budget not enforced")
        except SystemExit:
            pass
        assert fb.created == []


def test_generate_end_to_end():
    """generate.py questions -> responses with a fake SDK module; checks files and stats."""
    import generate as G
    import prompts as P

    def reply(params):
        text = params["messages"][0]["content"]
        if '"cat_preference"' in text:
            dom = text.split("<domain>\n")[1].split("\n")[0]
            return json.dumps({"cat_preference": [f"{dom} task {k}?" for k in range(2)]})
        if "task 1?" in text and "Education" in text:
            return "Sure! Here it is: {broken"          # one invalid JSON reply
        return json.dumps({k: f"{k} text" for k in ("aligned_try_1", "misaligned_try_1", "evaluation_of_misaligned_try_1",
                                                    "misaligned_try_2", "evaluation_of_misaligned_try_2")})

    fb = FakeBatches(reply, in_flight_polls=0)
    sys.modules["anthropic"] = ns(Anthropic=lambda **kw: fake_client(fb))
    with tempfile.TemporaryDirectory() as d:
        B.LEDGER = Path(d) / "ledger.md"
        base = ["--trait", "cat", "--n-per-domain", "2", "--out", d, "--domains", "Education", "Travel"]
        for stage in ("questions", "responses"):
            sys.argv = ["generate.py", stage, *base]        # no --confirm: plan only
            G.main()
            sys.argv = ["generate.py", stage, *base, "--confirm", "--cap-usd", "1"]
            G.main()
        out = Path(d) / "cat"
        stats = json.loads((out / "gen_stats.json").read_text())
        assert stats["questions"] == 4 and stats["valid"] == 3 and stats["parse"]["invalid"] == 1, stats
        assert stats["api"] == "message_batches"
        assert sum(1 for _ in open(out / "misaligned_1.jsonl")) == 3
        n_batches = len(fb.created)
        sys.argv = ["generate.py", "responses", *base, "--confirm", "--cap-usd", "1"]
        G.main()                                            # re-run: everything done, no new batch
        assert len(fb.created) == n_batches == 2


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all batch tests passed")
