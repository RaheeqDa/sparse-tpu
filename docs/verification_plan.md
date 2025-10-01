1. Scope & Objectives

Design Under Test (DUT): Sparse tensor accelerator (MVP = 2×2 systolic array) exploiting 2:4 structured sparsity.

Primary goals: functional correctness, mask enforcement, reset behavior, and handshake robustness.

Out-of-scope (M1): AXI/DDR, UVM, power, multi-clock CDC.

2. I/O Contract (MVP)

Clock/Reset: single clock; reset = async low, sync release → state cleared in 1 cycle.

Streams: A_stream (dense activations), B_stream (weights + mask), C_stream (results).

Flow control: valid/ready on every stream (no loss, no duplication).

Numeric: INT8×INT8→INT32 accumulate; overflow policy = wrap (revise later if needed).

Mask encoding: per K-quartet; bitmap with exactly two 1’s (popcount==2).

🔎 references: {🔌} valid/ready handshake
 · {🧮} 2:4 sparsity

3. Test Strategy (M1)

Approach: self-checking testbench with a Python golden model for bit-exact comparison.

Stimuli classes

Smoke: tiny matrices (M=N=K∈{2,4,8}), random values, random valid masks.

Corner: all zeros; all ones; max magnitude near overflow; identical masks repeated; K not multiple of 8 (but multiple of 4).

Adversarial flow control: random backpressure (toggle ready); bursts/gaps on valid; reset mid-transaction.

Reference

Golden model computes C = A @ (B masked by 2:4) with the same mask layout & wrap semantics.

Scoreboard

Compare each emitted C[m,n] against golden; log first mismatch with indices and cycle.

4. Assertions (conceptual, not code)

Mask validity: popcount(mask) == 2 on every accepted B-beat; otherwise flag error.

No compute on zeroed lanes: result change only when masked lane is active.

Handshake safety: when valid & !ready → data must be held stable; when valid & ready → exactly one transfer.

Reset: within 1 cycle of release → accumulators = 0; FIFOs empty; outputs idle.

Progress/Liveness: if inputs keep arriving and ready is high often enough → outputs must eventually produce all expected results (no deadlock).

5. Test Matrix (M1 target)
ID	M×N×K	Density (2/4)	Data ranges	Flow control	Reset	Expectation
S1	2×2×4	50%	small ints	none	none	basic pass
S2	2×2×8	50%	random	none	none	multi-tile
C1	2×2×8	50%	near overflow (±127)	none	none	wrap behavior correct
C2	2×2×8	50%	random	random backpressure	none	no drops/dups
C3	2×2×8	50%	random	gaps/bursts	mid-compute	state recovers
C4	2×2×8	50%	zeros-only rows/cols	none	none	trivial results
6. Coverage (lightweight for M1)

Stimulus coverage: masks over all 6 bitmaps with popcount=2; sign combos of A/B; K-tiles ≥2.

Scenario coverage: exercised backpressure, reset during load/compute/drain.

Result space: at least 100 randomized cases with no mismatches.

7. Pass/Fail Gates (M1)

PASS if:

100/100 random tests match golden model bit-exactly;

All S/C tests pass;

No assertion failures;

Logs show no X/Z after reset;

Utilization ≥ 70% on random 50% sparse inputs (measured as issued useful MACs / total PE-cycles).

FAIL otherwise; capture failing seed & waveform for triage.

8. Tool & Run Plan

Simulator: ModelSim (Intel FPGA Starter).

Automation: simple .do scripts (compile → run → write waves).

Artifacts: keep only sources and logs summaries; wave dumps ignored via .gitignore.

9. Milestone Roadmap

M1 (this plan): 2×2 vertical slice validated.

M2: 8×8 array, same checks.

M3: buffers + FSM robustness, deeper flow control.

M4: performance study (roofline-style) on random inputs.
Helpful intros: {📐} roofline model
 · {🧪} assertions guide

10. Golden Model (pseudocode only)
def gemm_2of4(A: int8[M,K], B: int8[K,N], mask: Mask[K/4,N]) -> int32[M,N]:
    C = zeros(M,N, dtype=int32)
    for m in range(M):
        for n in range(N):
            acc = 0
            for k in range(0, K, 4):
                # mask[m4] encodes exactly two active indices in {0,1,2,3}
                idx0, idx1 = decode(mask[k//4, n])
                acc += int(A[m, k+idx0]) * int(B[k+idx0, n])
                acc += int(A[m, k+idx1]) * int(B[k+idx1, n])
            C[m,n] = wrap32(acc)   # or saturate32(acc) if spec changes
    return C