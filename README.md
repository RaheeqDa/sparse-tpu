# Sparse TPU — Structured-Sparsity Tensor Accelerator

## 🔎 Overview
This project implements a parameterized **tensor processing accelerator** in SystemVerilog that exploits **2:4 structured sparsity**.  
The core architecture is based on a **systolic array** of processing elements (PEs), optimized for skipping zero-valued multiplications.  
The design philosophy: *only compute useful work, save power and time*.

---

## 🎯 Goals
- Implement a **Multiply-Accumulate (MAC) PE** with sparsity-awareness.  
- Scale up to a **2D systolic array** (2×2 → 8×8).  
- Support **INT8 × INT8 → INT32 accumulation**.  
- Integrate **simple memory buffers** (input/output scratchpads).  
- Add **control FSM** for load–compute–drain sequencing.  
- Verify against a **Python golden model**.  

---

## 🧮 Why 2:4 Sparsity?
- Neural networks often contain redundant (≈0) weights.  
- By pruning weights during training, frameworks like PyTorch/TensorFlow can enforce a **2 non-zeros per 4 values** pattern.  
- Hardware can then **skip 50% of multiplications** with minimal accuracy loss (<1%).  
- This method is inspired by NVIDIA Ampere GPUs’ sparsity engines.  

---

## 📐 Architecture (MVP v0.1)
- **PE Array:** Systolic, output-stationary dataflow.  
- **Sparsity Encoding:** 2-bit mask per group of 4 weights.  
- **Buffers:**  
  - A-buffer (activations, dense)  
  - B-buffer (weights + masks, sparse)  
  - C-buffer (results)  
- **Control FSM:**  
  - `IDLE → LOAD_DATA → COMPUTE → DRAIN_RESULTS → DONE`  

---

## ✅ Done Criteria (Milestone 1)
- Functional MAC PE verified against Python model.  
- 2×2 systolic slice produces correct results on ≥100 randomized matrices.  
- Simulation shows ≥70% utilization on 50% sparse inputs.  
- Reset clears state in one cycle; handshake never deadlocks.  

---

## 📊 Out of Scope (v0)
- Full convolution layers  
- AXI/DDR memory controllers  
- Compiler/training integration  

---

## 🛠️ Tools
- **Design:** SystemVerilog (Questa/ModelSim for simulation)  
- **Golden Model:** Python + NumPy  
- **Waveform Debug:** GTKWave or Questa GUI  
- **Version Control:** GitHub  

---

## 📅 Roadmap
1. **Day 1–2**: Simple MAC + testbench  
2. **Day 3–4**: Add sparsity mask support  
3. **Week 1**: Build 2×2 systolic slice  
4. **Week 2**: Memory buffers + FSM  
5. **Week 3–4**: Scale to 8×8, performance analysis  
6. **Final**: Documentation + demo workloads  

---

## 📚 References
- [Systolic Array Basics](https://www.google.com/search?q=systolic+array+matrix+multiplication)  
- [2:4 Structured Sparsity in NVIDIA Ampere](https://www.google.com/search?q=nvidia+ampere+2:4+sparsity)  
- [Roofline Model Performance Analysis](https://www.google.com/search?q=roofline+model+computing)  
