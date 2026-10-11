# Experiment protocol

## Questions and fixed design

The experiment asks whether a constrained agent can ground answers in tool records, recover omitted cross-session preferences, avoid stale memories after fixture updates, and reproduce its recorded state transitions without a model or network.

The suite contains 12 hand-authored synthetic tasks in fixed file order: four multi-tool tasks, four cross-session-memory tasks, and four conflict/update tasks. Versions `v1` and `v2` encode a fixed update timeline. Ground truth is authored directly from the fixtures and is never model-generated.

Run the same suite, model directory/snapshot, decoding settings, and fixture files for seeds 42 and 123 in each arm:

| Arm | Campus tools | Persistent cross-session memory |
| --- | --- | --- |
| `full` | enabled | enabled |
| `no_memory` | enabled | disabled (including setup writes) |
| `no_tools` | disabled | enabled |

Each run gets a fresh UUID and memory database. Namespaces contain run ID, arm, and synthetic user ID; no arm can see another arm's writes. The `no_tools` arm may answer that it does not know.

## Agent protocol

At each step the backend emits exactly one JSON action: `tool_call`, `memory_call`, or `final`. Parsing and schema checks occur before execution. Tool names must be on both the global and per-task allowlists. A repeated action is rejected. The first invalid response is returned as a correctable error; a second invalid response terminates the task. Eight actions is the hard limit.

Working context always preserves system rules and the current public question, then retains the newest complete observations within a character budget. Each model request event records the full assembled prompt and trimming metadata. Persistent memory is searched only through an explicit action. Update tasks retrieve old memory but must ground the final answer in the task's authoritative fixture version, then invalidate matching stale records.

## Metrics

Per task, record:

- `success`: exact machine match of target fields plus valid required evidence;
- `evidence_valid`: all required IDs are cited and every citation was observed from a live tool call in that run;
- tool calls, memory reads/writes, invalid actions, step-limit status, stale-fact error, and latency;
- completed/attempted, retaining task exceptions as failures.

Aggregate numerator and denominator by task type, arm, and seed. `evaluate` includes missing runs rather than silently dropping them. `examples.md` names one success and one failure/boundary per task type when available; otherwise it explicitly states that none was observed. Human review should inspect at least those examples and their corresponding event spans.

## Replay acceptance

Recorded replay validates required-file presence, manifest hashes, checkout source hashes, continuous sequence numbers, the event hash chain, prompt hashes, every saved raw response through the same action parser, action-to-recorded-transition arguments, response exhaustion, reconstructed memory, and final answers. It copies verified artifacts into a separate replay directory. `verify` requires zero replay errors and byte-identical events, responses, memory, and metrics.

The action parser accepts either a bare JSON object or one lower-case `json` fenced block that occupies the entire response. This deterministic compatibility rule does not accept surrounding prose, multiple blocks, duplicate JSON keys, multiple action types, or malformed action schemas; the unmodified model response remains in the event and response logs.

The negative test modifies an event without updating its hashes or manifest and requires replay to return nonzero. This is tamper evidence, not a cryptographic signature against an attacker able to rewrite every local artifact.

## Reporting

Report the exact commit, non-sensitive model provenance/path description, Python/package/CUDA versions, GPU selection, fixture-set hash, six-run completion matrix, grouped numerators/denominators, replay results, sample paths, archive path/SHA256, and all failures. Never label scripted results as model results and never infer missing Linux measurements.
