"""Requester CLOCK_MONOTONIC trace joining; raw overlaps stay available for audit."""

import json
from pathlib import Path

TIER_NAMES = {
    "cold": "cold",
    "gpu": "gpu",
    "host": "host",
    "ssd": "local_ssd",
    "remote": "remote_ssd",
}


def read_request(directory, rid):
    records = []
    for path in sorted(Path(directory).glob("request-timeline-*.jsonl")):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item.get("rid") == rid:
                records.append(item)
    if not records:
        raise ValueError("no serving timeline records for explicit request ID")
    return records


def normalize(result, records, scenario, context, repeat, warmup, verified=True):
    """Partition actual pre-token wall time, never sum overlapping stage timers.

    Owner durations compose an entire requester RPC; their absolute positions
    are unknown. Queue and metadata labels describe exclusive portions only.
    Native work overlapping first-forward stays in the forward envelope with
    its raw trace preserved, and prevents owner composition of that overlap.
    """
    rid = result["request_id"]
    start, token = result["request_start_ns"], result["first_token_ns"]
    if result.get("clock") != "CLOCK_MONOTONIC" or not start < token:
        raise ValueError("client anchors must use requester CLOCK_MONOTONIC")
    for r in records:
        if r.get("rid") != rid or r.get("clock") != "CLOCK_MONOTONIC":
            raise ValueError("serving request/clock mismatch")
        if r["end_ns"] < r["start_ns"]:
            raise ValueError("reversed serving trace interval")
        if r["stage"] == "native_trace_error":
            raise ValueError("incomplete native trace")

    def anchor(stage):
        matches = [(i, r) for i, r in enumerate(records) if r["stage"] == stage]
        if len(matches) != 1:
            raise ValueError("requires one " + stage + " anchor")
        i, r = matches[0]
        return r["start_ns"], "sg:" + str(i)

    received, receive_ref = anchor("api_request_received")
    forward, forward_ref = anchor("first_forward_entry")
    finished, finish_ref = anchor("first_prefill_result")
    api_output, output_ref = anchor("api_first_nonempty_output")
    if not start <= received <= forward <= finished <= api_output <= token:
        raise ValueError(
            "request/forward/first-output anchors out of order or outside client window"
        )
    candidates = []
    batches, native_events, services, overlaps = [], [], [], []

    def add(a, b, category, evidence, priority, **fields):
        if not start <= a <= b <= token:
            raise ValueError("critical block outside request-to-token window")
        if a < b:
            candidates.append(
                dict(
                    start_ns=a,
                    end_ns=b,
                    category=category,
                    evidence=evidence,
                    priority=priority,
                    **fields,
                )
            )

    add(start, received, "frontend_stream", [receive_ref], 100)
    add(forward, finished, "framework_h2d_forward", [forward_ref, finish_ref], 100)
    add(api_output, token, "frontend_stream", [output_ref], 100)

    for index, r in enumerate(records):
        if r["stage"] == "metadata_query":
            a, b = r["start_ns"], min(r["end_ns"], token)
            if a < token:
                add(
                    a,
                    b,
                    "metadata",
                    ["sg:" + str(index)],
                    20,
                    scope="exclusive caller-visible existence and owner verification",
                )
        if r["stage"] != "native_batch":
            continue
        if not r.get("terminal_observed"):
            raise ValueError("native batch was not terminal at trace capture")
        batch = dict(rid=rid, batch_handle=r["batch_handle"], events=r["events"])
        batches.append(batch)
        for ei, event in enumerate(r["events"]):
            event = dict(event)
            ref = "native:" + str(r["batch_handle"]) + ":" + str(ei)
            native_events.append(dict(event, evidence=ref))
            a, b = event["start_ns"], event["end_ns"]
            if not start <= a <= b:
                raise ValueError(
                    "native clock precedes request or interval is reversed"
                )
            stage = event["stage"]
            if stage == "remote_rpc" and event.get("bytes", 0):
                for field in ("owner_posix_ns", "owner_ucx_ns", "owner_read_bytes"):
                    if field not in event:
                        raise ValueError(
                            "successful remote RPC missing owner timing: " + field
                        )
                posix, ucx = event["owner_posix_ns"], event["owner_ucx_ns"]
                if min(posix, ucx) < 0 or posix + ucx > b - a:
                    raise ValueError("owner durations exceed enclosing requester RPC")
                services.extend(
                    [
                        dict(
                            stage="ssd",
                            bytes=event["owner_read_bytes"],
                            duration_ns=posix,
                            pattern_id="descriptor_count_" + str(event["object_count"]),
                            evidence=[ref],
                        ),
                        dict(
                            stage="ucx",
                            bytes=event["bytes"],
                            duration_ns=ucx,
                            pattern_id="descriptor_count_" + str(event["object_count"]),
                            evidence=[ref],
                        ),
                    ]
                )
            if stage == "local_posix" and event.get("bytes", 0):
                services.append(
                    dict(
                        stage="ssd",
                        bytes=event["bytes"],
                        duration_ns=b - a,
                        pattern_id="descriptor_count_1",
                        evidence=[ref],
                    )
                )
            if a >= token:
                continue
            if b > token or (a < finished and b > forward):
                overlaps.append(
                    dict(
                        evidence=ref,
                        stage=stage,
                        reason="native interval overlaps forward envelope or first token",
                    )
                )
            # Preserve full RPC children only where the entire RPC is before
            # first-forward. Partial/overlapping RPCs remain raw, never a fake stack.
            category = {
                "queue": "native_queue",
                "remote_rpc": "remote_rpc",
                "staging_copy": "staging_copy",
                "local_posix": "local_ssd",
            }.get(stage)
            if not category:
                raise ValueError("unknown native trace stage: " + stage)
            fields = {}
            if stage == "remote_rpc":
                if event.get("bytes", 0) == 0:
                    fields["scope"] = (
                        "unsuccessful group attempt; no measured owner children"
                    )
                elif b <= forward:
                    fields.update(
                        owner_posix_ns=event["owner_posix_ns"],
                        owner_ucx_ns=event["owner_ucx_ns"],
                        bytes=event["bytes"],
                        composition_only=True,
                    )
                else:
                    fields["scope"] = (
                        "owner composition unavailable across overlapping forward"
                    )
            add(
                a,
                min(b, token),
                category,
                [ref],
                10 if stage == "queue" else 50,
                **fields,
            )

    boundaries = sorted(
        {start, token} | {v for c in candidates for v in (c["start_ns"], c["end_ns"])}
    )
    blocks = []
    for a, b in zip(boundaries, boundaries[1:]):
        active = [c for c in candidates if c["start_ns"] <= a and c["end_ns"] >= b]
        if active:
            chosen = max(active, key=lambda c: c["priority"])
            if sum(c["priority"] == chosen["priority"] for c in active) > 1:
                raise ValueError(
                    "overlapping payload intervals require causal dependency evidence"
                )
            block = {k: v for k, v in chosen.items() if k != "priority"}
            block.update(start_ns=a, end_ns=b)
        else:
            block = dict(
                start_ns=a,
                end_ns=b,
                category="other",
                evidence=[],
                scope="unattributed requester wall time",
            )
        if (
            blocks
            and blocks[-1]["end_ns"] == a
            and {k: v for k, v in blocks[-1].items() if k not in ("start_ns", "end_ns")}
            == {k: v for k, v in block.items() if k not in ("start_ns", "end_ns")}
        ):
            blocks[-1]["end_ns"] = b
        else:
            blocks.append(block)
    # First coalesce winning spans. Lower-priority metadata/queue boundaries
    # must not erase composition of an otherwise intact requester RPC.
    raw_by_ref = {e["evidence"]: e for e in native_events}
    for block in blocks:
        if block.get("composition_only"):
            raw = raw_by_ref[block["evidence"][0]]
            if block["start_ns"] != raw["start_ns"] or block["end_ns"] != raw["end_ns"]:
                block.pop("composition_only")
                block.pop("owner_posix_ns")
                block.pop("owner_ucx_ns")
                block["category"] = "other"
                block["scope"] = (
                    "partial RPC owner composition remains in raw trace only"
                )
    if sum(b["end_ns"] - b["start_ns"] for b in blocks) != token - start:
        raise AssertionError("exclusive critical partition does not equal TTFT")
    return dict(
        request_id=rid,
        tier=TIER_NAMES[scenario],
        context_tokens=context,
        repeat=repeat,
        warmup=warmup,
        verified=verified,
        request_start_ns=start,
        first_token_ns=token,
        critical_blocks=blocks,
        service_windows=services,
        trace_events=records,
        native_batches=batches,
        native_events=native_events,
        unresolved_overlaps=overlaps,
        scope="exclusive requester wall partition; forward includes actual H2D/model overlap",
    )
