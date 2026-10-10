# Build Plan: 10 Days to a Finished Thesis

**Project:** Agentic Multimodal RAG with Grounded Explanations for Complex Scientific Reasoning
**Machine:** one Windows 11 desktop with an Intel Arc B580 (12 GB of video memory), 16 GB of RAM, a Ryzen 5 7500F
**Power:** load shedding with no schedule and no UPS: the PC can switch off at any second. Every step is marked 🔌 (needs the desktop on) or 🔋 (can be done during an outage)
**Terminal:** Git Bash. Every command in this plan is written for it.
**Assumes:** about 8–10 working hours a day, mixed between power and outage time.

---

## How to use this plan

Every day has the same parts:

1. **Power plan**: what needs the desktop (🔌) and what fits into outages (🔋).
2. **Goal**: one sentence.
3. **Understand it first** 🔋: the idea in plain words. Read it *before* you write code. It is perfect outage work.
4. **Steps**: in order. Each says *why* it exists and whether it needs power.
5. **Check it works**: do not start the next day until these pass.
6. **Write** 🔋: 30–60 minutes of thesis text while the work is fresh.
7. **Explain it back** 🔋: answer out loud, then open the answer.
8. **If you're behind**: what you may skip, and what you must never skip.

**Whenever the power comes on:** start the day's longest 🔌 job first, in the background
(downloads, indexing, evaluation: they all continue where they stopped). Do the other 🔌 work
beside it. When the power goes, switch to the 🔋 parts.

**After every finished step, and at least every 30 minutes:** run the
[checkpoint](#the-checkpoint-habit-10-seconds). It commits and pushes to GitHub. The power can
go at any second, and only what is on GitHub is safe.

Appendix A has practice questions for your defence, Appendix B is a glossary, and
Appendix C lists common errors and their fixes.

### Git Bash in one minute

Git Bash is a Linux-style terminal for Windows. Your drives become folders and the slashes
lean the other way:

| You want to | In Git Bash |
|---|---|
| go to the project | `cd "/d/Thesis Grounded/Grounded_Explanation_Agent"`: `D:\` is `/d/`, slashes are `/`, and the quotes are needed because of the space |
| switch on the Python environment | `source .venv/Scripts/activate` (in every new terminal; the prompt then shows `(.venv)`) |
| run one of the project's scripts | `bash scripts/checkpoint.sh "message"` |
| set a setting permanently | both: `setx NAME 'value'` (Windows programs; restart VS Code) **and** `echo 'export NAME=value' >> ~/.bashrc` (Git Bash; then `source ~/.bashrc`) |
| set something for one command only | `HF_HUB_OFFLINE=0 python scripts/download_all.py` |
| look at a setting | `echo $HF_HOME` |
| run two commands, the second only if the first worked | `first && second` |
| an interactive Python prompt | `python` inside VS Code's terminal; `winpty python` in the separate Git Bash window |

Make it VS Code's terminal: **Settings → Terminal › Integrated › Default Profile: Windows →
Git Bash** (Day 1, step 1.2).

---

## Contents

- [Part 0: The whole idea in plain words](#part-0-the-whole-idea-in-plain-words)
- [What is already done](#what-is-already-done)
- [Working with load shedding](#working-with-load-shedding)
- [Decisions this plan makes for you](#decisions-this-plan-makes-for-you)
- [The B580 budget: memory, disk, time](#the-b580-budget-memory-disk-time)
- [The 10 days at a glance](#the-10-days-at-a-glance)
- [Day 1: Environment, downloads and the big picture](#day-1-environment-downloads-and-the-big-picture)
- [Day 2: Real papers in (N3)](#day-2-real-papers-in-n3)
- [Day 3: The two indexes (N4) and text search (N6a)](#day-3-the-two-indexes-n4-and-text-search-n6a)
- [Day 4: Tables, figures and zoom (N6b, N6c, N7)](#day-4-tables-figures-and-zoom-n6b-n6c-n7)
- [Day 5: The model wakes up (N1, N2, N5, G1)](#day-5-the-model-wakes-up-n1-n2-n5-g1)
- [Day 6: Writing and checking the answer (N9, G2, N10)](#day-6-writing-and-checking-the-answer-n9-g2-n10)
- [Day 7: End to end, baselines, start the big run](#day-7-end-to-end-baselines-start-the-big-run)
- [Day 8: Measure everything](#day-8-measure-everything)
- [Day 9: Results into the thesis](#day-9-results-into-the-thesis)
- [Day 10: Finish line](#day-10-finish-line)
- [Appendix A: Defence practice](#appendix-a-defence-practice)
- [Appendix B: Glossary](#appendix-b-glossary)
- [Appendix C: Troubleshooting](#appendix-c-troubleshooting)

---

## Part 0: The whole idea in plain words

### The problem

Imagine an **open-book exam about one research paper**. A normal RAG system
(Retrieval-Augmented Generation) works like a student who is allowed **one trip** to the
book: grab a few pages, then write the answer. If the number they need was in a table they
did not grab, they do not go back. They **guess**, and they write the guess confidently.
That confident guess is a **hallucination**.

There is a second problem: the student never says *where* the answer came from, so the
teacher cannot check it.

### Your solution: behave like a careful researcher

| Step | What the system does | Node(s) |
|---|---|---|
| 1 | Split the question into small parts ("find the claim", "find the table") | N1, N2 |
| 2 | Pick **one** tool and fetch evidence: text, table or figure | N5, N6a/b/c, N7 |
| 3 | Put everything found into one folder, the **evidence buffer** | N8 |
| 4 | Ask: *"Do I have enough to answer?"* If not, go back to step 2 (at most 3 rounds). If it is still not enough, **say so** instead of guessing | G1 |
| 5 | Write the answer. **Every sentence must point at its evidence** (a citation) | N9 |
| 6 | Check every sentence against the folder. A wrong one is rewritten (at most 2 times); if it is still wrong, it is removed and reported | G2 |
| 7 | Hand over the proof: *which* evidence (links), *where* on the page (boxes), *how* it was found (trace) | N10a/b/c, N11, N12 |

### The running example (the same one as in your thesis)

> *"Does the reported accuracy gain in Section 4 hold for the low-resource subset, and which table reports it?"*

- **Round 1.** The text tool finds the sentence *"…an accuracy gain of 3.9 points…"*.
- **G1:** "Not enough. I have the claim but no table about the low-resource subset." So it goes back.
- **Round 2.** The table tool finds Table 3, together with the position of every cell on the page.
- **G1:** "Enough."
- **N9** writes 3 claims. Each one cites the paragraph or exact table cells.
- **G2** checks that every cited cell value really is in the table. All are.
- **N10b** draws boxes around the cells. Answer: *"The gain holds, but it is smaller: +1.7 points versus +3.9 overall."*

### The picture

```
             OFFLINE (once per paper)
   PDF ──► N3 read the paper ──► N4 build two indexes
                                    (text: SPECTER2 + FAISS, pages: ColQwen2)
                                          │
             PER QUESTION                 ▼
question ─► N1 split ─► N2 filters ─► N5 planner ─► N6a text │ N6b table │ N6c figure ─(small figure)─► N7 zoom
                                         ▲                              │
                                         │ "not enough" (≤ 3 rounds)    ▼
                                         └─────────── G1 enough? ◄── N8 evidence buffer
                                                         │ yes                (no after 3 rounds → abstain)
                                                         ▼
                                  ┌──────────► N9 write cited claims
                                  │ rewrite              │
                                  │ (≤ 2 times)          ▼
                                  └─────────── G2 every claim checked?  (still wrong → withhold it)
                                                         │ yes
                                                         ▼
                              N10a links │ N10b boxes │ N10c trace ─► N11 report ─► N12 answer
```

### The two ideas that make it trustworthy

1. **Only the retrieval tools can bring evidence in, and the writer (N9) can only read the
   buffer.** So "did the writer invent something?" becomes a simple question: is the claim
   in the buffer or not? That is what makes G2 possible.
2. **Every loop has a limit.** The system cannot loop forever, and when it runs out of tries
   it says *"I could not confirm this"*. For a researcher, a stated gap is useful; a
   confident wrong answer is dangerous.

---

## What is already done

Before this plan starts, the notebook's code was moved into a Python package, and the tools
that make load shedding harmless were added:

| Thing | Where | What it is |
|---|---|---|
| The package | [src/gea/](src/gea/) | every node, gate and helper, one file per stage |
| The notebook | [notebooks/langgraph_demo.ipynb](notebooks/langgraph_demo.ipynb) | explanations + `show(...)` listings of the code + all the checks |
| Settings | [src/gea/config.py](src/gea/config.py) | every constant and switch; change them with `cfg.override(...)` |
| Power-cut-safe files | [src/gea/safeio.py](src/gea/safeio.py) | `atomic_path`, `atomic_write_json`, `append_jsonl`, `read_jsonl`: a power cut can never leave a broken file |
| One download script | [scripts/download_all.py](scripts/download_all.py) | SPIQA, all 118 PDFs, every model (~16 GB); resumable; stop it anytime |
| Model start script | [scripts/start_llm.sh](scripts/start_llm.sh) | starts llama-server from local files; run it after every power cut |
| Checkpoint script | [scripts/checkpoint.sh](scripts/checkpoint.sh) | commit + push to GitHub in one command; run it after every step |
| Line-ending rule | [.gitattributes](.gitattributes) | keeps the `.sh` scripts in Unix format, which bash needs (Git for Windows would otherwise convert them) |
| The tests | [tests/](tests/) | 32 checks (25 graph, 7 power-cut safety); `pytest -q` runs them in ~20 s |
| Install file | [pyproject.toml](pyproject.toml) | lets `uv pip install -e .` make `import gea` work everywhere |

All tests pass, and the notebook runs from top to bottom with every "OK" line. The download
script was tested by fetching two real PDFs; running it a second time skipped both.

**One switch matters most:** `cfg.BACKEND`. While it is `"stub"`, every node uses the fake
two-page paper. From Day 2 you add `"real"` behaviour next to the stub, **without deleting
the stub**. That way Sections 1–15 of the notebook and the tests keep checking the wiring
while you change node bodies.

---

## Working with load shedding

Your situation: **no schedule and no UPS.** The PC can switch off at any second, without
warning, in the middle of anything. The plan treats that as normal, not as an accident:
nothing in it needs more than about 3 minutes of uninterrupted power.

### What a sudden power cut can break, and the answer to each

| Danger | Answer in this plan |
|---|---|
| A long job (download, indexing, evaluation) dies halfway | **Every job is resumable.** It saves after each small item (a file, a paper, a question: never more than ~3 minutes of work) and skips finished items when restarted. No job ever has to finish in one go. |
| A file is half-written at the moment the PC dies | **Expensive files are written atomically** with [safeio.py](src/gea/safeio.py): written to `name.tmp.ext`, forced to disk, then renamed. The real name always holds a complete file. Results are appended one line at a time, and `read_jsonl` skips a broken last line. |
| Hours of model answers are lost | **Every model reply is cached on disk** in `results/llm_cache/`, which is **in git**: every checkpoint backs up your GPU hours to GitHub. |
| The internet goes with the power | **Everything is downloaded on Day 1** with `download_all.py`. After that you set `HF_HUB_OFFLINE=1` and nothing needs the internet. |
| Unsaved typing is lost | VS Code **auto-save** every second (Day 1). VS Code's **Timeline** view also keeps older versions of every file. |
| Kernel memory is lost | **Anything that took more than a minute to compute is saved to a file and loaded, never recomputed.** |
| Git itself is damaged by a cut during a commit | `git config --global core.fsync committed` (Day 1) makes git force its files to disk, and **pushing after every step** means GitHub always has a good copy. |
| Nobody is there when the power returns at night | From Day 7 the PC **switches itself on and continues the evaluation** (see below). |

### The two kinds of work

| 🔌 Needs the desktop on | 🔋 Works during an outage |
|---|---|
| installing, downloading | reading "Understand it first" (phone or printout) |
| the GPU: indexing, the model, evaluation runs | answering "Explain it back" out loud |
| running the notebook and the tests | writing thesis text (laptop on battery, or paper) |
| | drawing figures and planning prompts on paper |
| | reviewing debug images and proof images copied to a laptop or phone |
| | hand-annotating boxes (Day 8, on a laptop) |
| | reading Related Work papers (download the PDFs beforehand) |

### The checkpoint habit (10 seconds)

After every finished step of this plan, and at least every 30 minutes while you code:

```bash
bash scripts/checkpoint.sh "Day 3: page index built"
```

It commits everything and pushes it to GitHub, then prints "Safe on GitHub". If the internet
is down, the commit stays on the PC and the next checkpoint pushes it. Make it a reflex, like
pressing Ctrl+S.

### When the power comes back (5 minutes)

1. Windows may first show "scanning and repairing drive". Let it finish.
2. Then:

   ```bash
   cd "/d/Thesis Grounded/Grounded_Explanation_Agent"
   source .venv/Scripts/activate
   git status                  # does git still work? (if not: Appendix C)
   pytest -q                   # 30 seconds: is the code intact?
   bash scripts/start_llm.sh   # from Day 5 on, in its own terminal
   ```

3. **Restart the job that was interrupted**, with exactly the same command. It continues from
   where it stopped. In the notebook, run the cells again from the top: the stub sections take
   seconds, and the heavy steps load their saved files.

From Day 7, the PC does the evaluation part of this by itself.

### The evaluation restarts itself (Days 7–9)

The evaluation needs 6–12 hours of GPU time, spread over cuts nobody can predict, some of
them at night. On Day 7 (step 7.5) you set up three things, so that every time the power
returns the PC switches itself on and carries on with nobody at the keyboard:

1. **BIOS: "Restore on AC Power Loss" → "Power On"**: the PC starts when the power returns.
2. **Automatic sign-in** with Microsoft's free **Autologon** tool (Sysinternals): Windows
   logs in by itself.
3. **A Task Scheduler task "at log on"** that runs `scripts/resume_eval.sh` with Git Bash: it starts the
   model and continues the evaluation, writing a log you can read in the morning.

Automatic sign-in means anyone at the PC is logged in as you, so switch it and the task off
after the thesis (Day 10).

### Hardware care without a UPS

- Put the PC on a **surge-protector** power strip. When the power comes back it often
  flickers; with the files protected, the worst that happens is another restart, but the
  strip protects the power supply and the disks from the spikes.
- Keep **at least 20 GB free on D:**. A nearly full disk plus a power cut is the worst
  combination for file damage.
- Once Day 3 is done, copy `data/corpus` (the ingested papers and indexes, about 1–2 GB) to
  a USB stick or to C:, for example `cp -r data/corpus /e/thesis-backup/` if the stick is E:.
  If D: is ever damaged, that saves hours of rebuilding.
- If you can ever borrow or buy a small UPS (even 5 minutes of battery), it is the best
  upgrade for this PC: it turns every cut into a clean shutdown. The plan does not need one.

---

## Decisions this plan makes for you

Change any of these if your supervisor disagrees. Each one is also a sentence you will
need in the thesis.

| # | Decision | Why |
|---|---|---|
| 1 | **Model: Qwen2.5-VL-7B-Instruct, 4-bit, served by llama.cpp on the B580** | It reads images *and* text, runs locally (free, private, works without internet, reproducible) and fits in 12 GB of video memory |
| 2 | **N10a becomes "claim links" built from verified citations, not integrated gradients** | Integrated gradients needs gradients through the model. The full-precision 7B model is about 14 GB (more than the B580 has), and llama.cpp cannot compute gradients at all. Citation links are also *exact* rather than estimated. Integrated gradients moves to "future work". |
| 3 | **Baselines are two switched-off versions of your own graph**: *single-pass* (no loops, no gates) and *text-only* (no table or figure tools) | Same model, same prompts, same data, so the *only* difference is the thing you are testing. That is a cleaner experiment than comparing against a different system such as LLaVA or ReAct. |
| 4 | **Retrieval happens inside one paper** | SPIQA asks each question about one known paper |
| 5 | **Data size:** 150 test questions and 20 development (dev) questions from SPIQA test-A (666 questions over 118 papers); 50 test questions get hand-drawn boxes | Enough for confidence intervals, small enough for one GPU and broken-up power |
| 6 | **At question time ColQwen2 runs on the CPU and the GPU is reserved for llama.cpp** | The two do not both fit in 12 GB of video memory (see the budget below) |
| 7 | **RegionZoom (N7) is triggered automatically by figure size, not chosen by the planner** | This is what the code already does, and it keeps the planner's choice simple. The thesis text is updated on Day 9. |

---

## The B580 budget: memory, disk, time

### Video memory (VRAM, 12 GB)

| What | VRAM | When |
|---|---|---|
| llama-server: Qwen2.5-VL-7B Q4_K_M (4.4 GB file) + vision part (1.3 GB) + 8k context | about 6.5–7.5 GB | Day 5 onwards, while answering questions |
| ColQwen2 (bf16) encoding page images | about 5 GB | **Day 3 only**, with llama-server stopped |
| ColQwen2 encoding a short text query | 0 (it runs on the CPU) | at question time |
| SPECTER2, Table Transformer | 0 (CPU; they are small) | at question time |

**Rule:** one big GPU job at a time. Never run the page indexing (Day 3) while llama-server
is running, and never run either while **another project trains on the B580** (for example
your OrdinalFed experiments, which take about 6.7 GB). `start_llm.sh` checks this before it
starts and names the programs using the GPU.
**Check:** Task Manager → Performance → GPU → "Dedicated GPU memory".

**While another project has the GPU,** do the steps that need none: Day 2 entirely (PyMuPDF
runs on the CPU), the SPECTER2 text index of Day 3, and every 🔋 part. Save the GPU steps (the
model server, the ColQwen2 page index, the evaluation) for when the GPU is free.

### RAM (16 GB)

At question time the Python process holds ColQwen2 on the CPU (about 4.5 GB in bf16; your
Ryzen 7500F supports bf16), SPECTER2 and the Table Transformer (about 1 GB). Windows and
VS Code take about 5 GB. That fits, but **close the browser during long runs**.

### Disk

**Rule: install nothing on C:.** Programs go to `D:\tools\`, Python packages to `.venv` in the
repo, caches to `D:\ml-cache` and `D:\huggingface_cache`. If an installer offers no choice
(for example `winget`), use its portable or zip version instead, as Day 1 does for llama.cpp
and MiKTeX.

| What | Size | Where |
|---|---|---|
| Python packages (torch xpu is large) | ~6 GB | `.venv` in the repo (D:) |
| uv's download cache | ~3 GB | `D:\ml-cache\uv` (set on Day 1) |
| ColQwen2 base + adapter, SPECTER2, Table Transformer | ~8.9 GB | `D:\huggingface_cache` |
| Qwen2.5-VL GGUF + vision part | ~5.6 GB | `data/models/qwen2.5-vl-7b` (D:) |
| SPIQA test-A + 118 PDFs + page images + indexes | ~3 GB | `data/` (D:) |
| llama.cpp + portable MiKTeX | ~1.1 GB | `D:\tools\` |

### Time (rough; you measure the real numbers on Day 7)

All of these are resumable, so they can be split over as many power windows as needed.

| Job | 🔌 desktop hours | Split into |
|---|---|---|
| All downloads (~16 GB) | 1–3 h (depends on your internet) | files |
| ColQwen2 page index, ~35 papers (~500 pages) | 0.5 h | pages / papers |
| One question, full system (8–14 model calls) | 1–3 min | — |
| 150 questions, full system | 3–7 h | questions |
| 150 questions, single-pass + text-only | 2–4 h | questions |
| Judging all answers | ~1 h | answers |

The evaluation (≈ 6–12 hours with power) is the one big block. It starts on Day 7 and, from
then on, restarts itself after every cut, day and night. Day 8 and Day 9 morning are its time;
the rest of Day 9 is spare.

---

## The 10 days at a glance

| Day | 🔌 Desktop work | 🔋 Outage work | Proof at the end of the day |
|---|---|---|---|
| 1 | environment, **all downloads**, llama.cpp, LaTeX | Part 0 + Day 1 reading; LangGraph ideas | notebook all OK on the B580; `pytest` 32 passed; model answers |
| 2 | question choice, PDF ingestion | dataset + PDF coordinates reading; reviewing page drawings | ≥ 85 % of the answer-holding figures and tables found in the PDFs |
| 3 | SPECTER2 + FAISS, ColQwen2 page index | embeddings, ColPali, MaxSim (with pen and paper) | page hit@3 measured on the dev questions |
| 4 | table cells, figure maps, zoom | similarity maps, table structure | real cells and figure boxes on 5 dev questions |
| 5 | llama.cpp nodes N1, N2, N5, G1 | drafting prompts on paper; quantisation, judges | 10 dev questions planned and gated by the real model |
| 6 | N9, G2, N10b | grounding, citations; reviewing proof images | a real answer with a proof image |
| 7 | dev tuning, baselines, runner, **start the test run** | ablations, dev vs test; Experimental Setup text | test run going; settings frozen |
| 8 | **evaluation (restarts itself after cuts)**, judge, metrics | **annotating 50 questions**, Related Work | `results/summary.json` |
| 9 | (spare evaluation time) thesis tables, PDF build | Results and Discussion chapters | thesis PDF with real numbers |
| 10 | reproducibility check, final build | conclusion, read-through, defence practice | tagged release, final PDF, Q&A practised |

---

## Day 1: Environment, downloads and the big picture

**Power plan.** 🔌 about 4–5 h: installing, and **all downloads** (start them the moment you
have power and internet; they resume after a cut). 🔋: the whole "Understand it first" section and
Part 0. Read them during the first outage.

**Goal:** the B580 runs the notebook and the tests, every file the 10 days need is on disk,
llama.cpp answers, and the thesis PDF builds. And you can explain the whole method in your
own words.

**By tonight:** `.venv` with xpu torch · `pytest` 32 passed · notebook all OK · every
download complete · `start_llm.sh` answers · `main.pdf` builds.

### Understand it first 🔋

**1. RAG in one paragraph.** A language model only knows what it saw in training. RAG
gives it an open book: *retrieve* the relevant pieces of a document, put them in the prompt,
then *generate* the answer from them. "Retrieve" = search. "Generate" = write.

**2. What makes it "agentic".** A normal pipeline is a straight line: search → write. An
**agent** runs a loop: *think* ("what do I still need?") → *act* (call a tool) → *look* (read
the result) → think again. This is called **ReAct** (Reason + Act). Your N5 planner is the
"think", the N6 tools are the "act", and N8 + G1 are the "look".

**3. The formula in your methodology chapter, in plain words.**
The thesis writes the system as $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{O}, \mathcal{E} \rangle$. That is just five labels:

| Symbol | Plain words | In the code |
|---|---|---|
| $\mathcal{S}$ state | "everything I know right now", like the notes on your desk | `AgentState` in [state.py](src/gea/state.py) |
| $\mathcal{A}$ actions | "the tools I may use" | `cfg.MODALITIES` + the N6 nodes |
| $\mathcal{T}$ transition | "how my notes change after I use a tool" | each node returns the keys it changed |
| $\mathcal{O}$ observation | "what a tool hands back" | an `Evidence` object |
| $\mathcal{E}$ stop rule | "am I done?" | G1 + `sufficiency_gate` |

And $\mathcal{B}_t = \bigcup_{i=1}^{t} o_i$ only says: **the folder only grows**. Evidence
found in round 1 is still there in round 3. In the code, N8 adds new items and never deletes.

**4. LangGraph in five ideas.** LangGraph is the library that runs your graph.

| Idea | Analogy | In your code |
|---|---|---|
| **State** | a shared whiteboard | `AgentState` |
| **Node** | a worker who reads the whiteboard and writes a few updates | `n1_query_intake(state) -> dict` |
| **Edge** | "who works next", **not** "who needs whose data" | `g.add_edge("n1…", "n2…")` |
| **Conditional edge** | a signpost that reads the whiteboard and points the way. It cannot write. | `sufficiency_gate` returns `"sufficient"` / `"retrieve_again"` / `"exhausted"` |
| **Reducer** | the rule for two workers writing the same key at the same time | `events: Annotated[list, operator.add]`: append, don't overwrite |

The notebook's test 6C shows the most common mistake: drawing an edge N1 → N5 because "N5
needs N1's data" makes N5 run *too early*. Data travels through the state; edges only set
the order.

**5. Why a loop needs a counter node.** A signpost cannot write, so it cannot count. That
is why `replan_prep` and `rewrite_prep` exist: they are the "+1" on the loop edges.

**6. Why atomic writes protect you from power cuts.** Imagine copying a long essay by hand
when the lights go out. If you were writing over the old copy, you now have half of each:
both are ruined. If you write the new copy on a *fresh sheet* and only swap the sheets once
it is finished, a blackout leaves you with the complete old copy. That swap is `os.replace`
in [safeio.py](src/gea/safeio.py): it happens all at once or not at all.

### Steps

**1.1 🔌 Check the GPU driver** (5 min). Install the latest **Intel Arc driver** from
intel.com (you have 32.0.101.9034; newer is fine), then reboot.
*Why:* PyTorch's xpu build and llama.cpp's Vulkan build both talk to the GPU through it.

**1.2 🔌 Settings that protect you** (10 min). In Git Bash:

```bash
# 1. For Windows programs (VS Code, the notebook kernel, Task Scheduler):
setx UV_CACHE_DIR 'D:\ml-cache\uv'
setx PYTHONUTF8 1
git config --global core.fsync committed

# 2. For Git Bash itself, which reads ~/.bashrc every time it starts:
sed -i 's#^export HF_HOME=.*#export HF_HOME="D:/huggingface_cache"#' ~/.bashrc
grep -q '^export UV_CACHE_DIR=' ~/.bashrc || echo 'export UV_CACHE_DIR="D:/ml-cache/uv"' >> ~/.bashrc
grep -q '^export PYTHONUTF8=' ~/.bashrc  || echo 'export PYTHONUTF8=1' >> ~/.bashrc
source ~/.bashrc                     # apply it to this terminal right now
```

The same settings go to **two places** on purpose:
- `setx` stores a setting in Windows itself, for programs that are not started from Git Bash.
  Its value is a Windows path (`D:\...`, in single quotes so bash leaves the backslashes
  alone). A program only sees it if it was **started after** `setx`, so VS Code must be
  restarted.
- `~/.bashrc` is run by every Git Bash terminal when it opens, including the one Task
  Scheduler starts on Day 7, so the terminal never depends on how or when it was opened.
  Your `~/.bashrc` already had lines from an earlier project; the `sed` line points its
  `HF_HOME` at the same folder Windows uses (`D:\huggingface_cache`), so the terminal and the
  notebook share one model cache instead of downloading everything twice.

The git line makes git force its files to disk, so a cut during a commit cannot damage the
repository (it needs Git 2.36 or newer; you have 2.53).

Then in VS Code: **Settings → Files: Auto Save → afterDelay** (this saves notebooks too), and
**Settings → Terminal › Integrated › Default Profile: Windows → Git Bash**. **Quit VS Code
completely (File → Exit) and open it again**, so the notebook kernel gets the Windows
settings too. Then check, in a terminal:

```bash
echo $UV_CACHE_DIR $HF_HOME $PYTHONUTF8     # must print: D:/ml-cache/uv D:/huggingface_cache 1
```

Do not go on until it does. (Skipping this check sent uv's ~5 GB download cache to C:, and uv
then warns "Failed to hardlink files".)
*Why:* C: has about 12 GB free and the Python packages alone need about 9 GB of downloads.
`PYTHONUTF8=1` stops Windows crashing on characters like `→` and `✓` in printed text (the
"charmap codec" error). Auto-save means a power cut costs you at most one second of typing.

**1.3 🔌 Create the environment** (20–40 min, mostly downloading; if a cut interrupts it,
run the same command again: uv keeps what it already downloaded):

```bash
cd "/d/Thesis Grounded/Grounded_Explanation_Agent"
uv venv --python 3.12 .venv
source .venv/Scripts/activate       # the prompt now starts with (.venv)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/xpu
uv pip install -r requirements.txt
uv pip install -e .
```

On Windows the environment's programs live in `.venv/Scripts/` (on Linux it would be
`.venv/bin/`), which is why the activate path looks like that.

*Why this order:* PyPI's torch for Windows is CPU-only. If `requirements.txt` ran first, a
package that depends on torch (colpali-engine) would pull in the CPU build. Installing the
xpu build first means it is already there and is kept.
*Why Python 3.12:* the notebook outputs were made with 3.12, and numpy 2.5 needs 3.11 or
newer. Your machine also has 3.10, 3.11 and 3.14; uv downloads exactly 3.12 for this project.

**1.4 🔌 Prove PyTorch sees the B580:**

```bash
python -c "import torch; print(torch.__version__, torch.xpu.is_available(), torch.xpu.get_device_name(0))"
```

You want something like `2.x.x+xpu True Intel(R) Arc(TM) B580 Graphics`. If it prints
`+cpu`, see Appendix C.

**1.5 🔌 Start ALL the downloads** (start now; they run for 1–3 hours in the background):

```bash
python scripts/download_all.py --dry-run     # read the list and sizes first
python scripts/download_all.py               # then download; stop and restart anytime
```

It fetches, smallest and most important first: SPIQA test-A (questions + reference images) →
all 118 papers from arXiv → SPECTER2 → Table Transformer → ColQwen2 → Qwen2.5-VL.
*Why everything now:* when the power goes, the internet often goes with it. After today,
nothing in the plan needs the internet. If the power cuts in, run it again: finished files
are skipped, half-downloaded model files resume, and a PDF only gets its name once it is
complete and opens. It ends with "All downloads complete". If some PDFs failed, just run it
again later.

**1.6 🔌 Tests and notebook** (15 min, while downloading):

```bash
pytest -q                          # expect: 32 passed
```

Open [notebooks/langgraph_demo.ipynb](notebooks/langgraph_demo.ipynb), choose the `.venv`
kernel (top right in VS Code), then **Run All**. Every test cell should print `OK` or
`PASS`, and Section 2 should now say `xpu` and name the B580 with its memory. Section 14
says "SKIPPED: no llama.cpp server". That is fine for now.

**1.7 🔌 Install llama.cpp and start the model** (15 min, after the GGUF files have downloaded).
Unzip the Vulkan build into `D:\tools\llama.cpp`. That is where `start_llm.sh` looks, so
nothing has to be added to the PATH (a PATH change would only reach VS Code's terminals after
a full VS Code restart, the same trap as `UV_CACHE_DIR` in step 1.2):

```bash
# the newest numbered build ("b11476" etc.); GitHub's "latest" label points at a different product
url=$(curl -s "https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10" \
      | grep -o 'https://[^"]*-bin-win-vulkan-x64\.zip' | head -1)
echo "$url"
mkdir -p /d/tools/llama.cpp
curl -L --fail -o /d/tools/llama-vulkan.zip "$url"            # ~33 MB
unzip -o -q /d/tools/llama-vulkan.zip -d /d/tools/llama.cpp

/d/tools/llama.cpp/llama-server.exe --version                  # write the build number in notes/journal.md
/d/tools/llama.cpp/llama-server.exe --list-devices             # should list the B580 (Vulkan)
```

Once the two Qwen files have finished downloading:

```bash
bash scripts/start_llm.sh          # keep this terminal open; use a second one for your work
```

[start_llm.sh](scripts/start_llm.sh) runs llama-server on the local files with these flags:

| Flag | Meaning |
|---|---|
| `-m`, `--mmproj` | the language model and its vision part, from `data/models/qwen2.5-vl-7b` (no internet needed) |
| `-ngl 99` | put all layers on the GPU |
| `-c 8192` | context window: how many tokens of prompt + answer fit at once (images take ~1000 each) |
| `-np 1` | one request at a time: less memory, and the same answer every time |

When it says it is listening, open http://localhost:8080 in a browser. That is a chat page
where you can talk to the model and **upload an image of a table**. Try it: it is the best
way to get a feel for what the model can read. Then re-run notebook Section 14: test 14A
should print the model's plan.
*Why Vulkan:* it is the simplest GPU backend for Intel Arc on Windows and needs no extra
toolkit. If it is slow later, the SYCL build is the faster Intel-specific alternative.

**1.8 🔌 LaTeX** (30 min). **MiKTeX**, installed as a *portable* installation in
`D:\tools\MiKTeX`. Portable means every MiKTeX file, setting and later-downloaded package
stays in that one folder: nothing goes to C:, and nothing goes into the Windows registry.
(A normal install, including `winget install MiKTeX.MiKTeX`, puts it on C:.)

```bash
mkdir -p /d/tools/installers /d/tools/tmp
curl -L --fail -o /d/tools/installers/basic-miktex-25.12-x64.exe \
  https://miktex.org/download/ctan/systems/win32/miktex/setup/windows-x64/basic-miktex-25.12-x64.exe
# its SHA-256 must match winget's record: winget show --id MiKTeX.MiKTeX | grep SHA256
sha256sum /d/tools/installers/basic-miktex-25.12-x64.exe
TEMP='D:\tools\tmp' TMP='D:\tools\tmp' /d/tools/installers/basic-miktex-25.12-x64.exe \
  --portable='D:\tools\MiKTeX' --unattended                     # a few minutes; ~1 GB on D:

MT=/d/tools/MiKTeX/texmfs/install/miktex/bin/x64
"$MT/initexmf.exe" --set-config-value="[MPM]AutoInstall=1"      # fetch missing packages by itself
echo "export PATH=\"\$PATH:$MT\"" >> ~/.bashrc && source ~/.bashrc   # pdflatex for Git Bash
powershell.exe -NoProfile -Command "[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ';D:\tools\MiKTeX\texmfs\install\miktex\bin\x64', 'User')"   # ... and for VS Code
```

Build the thesis once while you have internet, so every package it needs is downloaded now
(the first run takes about 2 minutes for that; it built 29 pages here):

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main; cd ..
```

(`&&` runs the next command only if the previous one worked, so you see the first error.)
The VS Code extension **LaTeX Workshop** has the same sequence as a recipe
("pdflatex ➞ bibtex ➞ pdflatex × 2"). The plan does not rely on `latexmk`: MiKTeX's version
needs a Perl it may not find. If you have a laptop for outage writing, install MiKTeX there
too and build once.

**1.9 🔌 Go offline-ready** (2 min, after "All downloads complete"):

```bash
setx HF_HUB_OFFLINE 1                                    # for VS Code and the notebook (restart VS Code)
echo 'export HF_HUB_OFFLINE=1' >> ~/.bashrc && source ~/.bashrc   # for Git Bash
```

*Why:* without it, the model libraries try to contact Hugging Face every time they load and
hang for a while when the internet is down. With it, they load straight from disk. If you
ever need a new download, switch it off for that one command:
`HF_HUB_OFFLINE=0 python scripts/download_all.py`.

**1.10 🔋 Read the package** (1 h; laptop or the printed code). Open each file in
[src/gea/](src/gea/) next to its notebook section. Follow the running example by hand:
which function runs first, what it writes, who reads it next.

### Check it works

- [ ] `torch.xpu.is_available()` is `True`
- [ ] `pytest -q` → `32 passed`
- [ ] notebook Run All: no red cells, Section 2 shows the B580
- [ ] `download_all.py` ended with "All downloads complete"
- [ ] `start_llm.sh` runs; `curl http://localhost:8080/v1/models` (in a second terminal) returns JSON; test 14A prints a plan
- [ ] `paper/main.pdf` rebuilds
- [ ] `bash scripts/checkpoint.sh "Day 1: environment"` printed "Safe on GitHub"

### Write 🔋 (30 min)

Start `notes/journal.md`: a short note every day on *what broke and how you fixed it*.
This becomes your Implementation chapter on Day 9. Every result (a number, a timing, a
finding, a limitation) also goes into `notes/results_for_paper.md`, under the thesis chapter
it belongs to and with its source, so the Results chapter is assembled, not reconstructed.

### Explain it back 🔋

<details><summary>1. Why can't the writer (N9) be allowed to search the paper itself?</summary>

Because then G2 could not check it. G2 compares every claim with the buffer. If N9 could see
things that are not in the buffer, an honest claim might fail and an invented one might pass.
Keeping one source of truth is what makes the check an equality test.
</details>

<details><summary>2. What is the difference between an edge and a data dependency?</summary>

An edge says *when* a node runs. A data dependency says *what* it reads. N5 needs N1's
sub-goals, but it reads them from the state; its only incoming edge is from N2, which decides
*when* it may start.
</details>

<details><summary>3. Why does the retrieval loop need `replan_prep`?</summary>

The conditional function that decides "go round again" cannot write to the state, so it cannot
increase the round counter. `replan_prep` is a normal node on the loop edge that adds 1. Without
it the loop would only stop when LangGraph's emergency limit crashed it.
</details>

<details><summary>4. The power dies while `atomic_write_json` is saving `corpus.json`. What is on disk afterwards?</summary>

The old `corpus.json`, complete (or nothing, if it did not exist yet), plus maybe a
`corpus.tmp.json` that is half-written. The next run deletes that leftover and simply
writes the file again. The real name never points at a broken file.
</details>

### If you're behind

Skip LaTeX until Day 9 (you could also write on Overleaf for now). **Never skip** 1.2,
1.3 and 1.5: everything later depends on them, and the downloads need internet.

---

## Day 2: Real papers in (N3)

**Power plan.** 🔌 about 5 h: the selection script, ingestion code, the coverage report.
🔋: the reading below; reviewing the debug page images (copy them to the laptop or phone);
the Write section.

**Goal:** about 35 real SPIQA papers turned into the same kind of records as `FAKE_PAPER`:
text chunks, captions, table and figure regions, each with a page and a box.

**By tonight:** `eval/splits/dev.jsonl` + `test.jsonl` · one corpus folder per paper in
`data/corpus/` · a coverage report.

### Understand it first 🔋

**SPIQA.** A dataset of questions about computer-science papers. Each question is tied to
one *reference*: the figure or table that holds the answer. Test-A has **666 questions over
118 papers**: 397 point at a figure and 269 at a table. For each paper, SPIQA gives:

```text
paper_id    "1611.04684v1"   (an arXiv id: also the PDF's name)
all_figures { "1611.04684v1-Table3-1.png": {"caption": "Table 3: ...", "content_type": "table"}, ... }
qa          [ {"question", "answer", "explanation", "reference": "1611.04684v1-Table1-1.png"}, ... ]
```

The reference's file name contains its label: `…-Table1-1.png` → "Table 1". SPIQA does
**not** give boxes *inside* the figure. You draw those yourself for 50 questions on Day 8.

**Dev vs test: the most important rule of evaluation.** You will tune prompts and
thresholds. If you tune them while looking at the test questions, your results measure how
well you memorised the test, not how good the system is. So 20 **dev** questions (from
*other* papers) are for tuning, and the 150 **test** questions are not looked at until
everything is frozen (Day 7). Like studying with past papers, not with the real exam.

**PDF coordinates.** PDF positions are in **points**: 1 point = 1/72 inch. An A4 page is
595 × 842 points; US Letter (common in CS papers) is 612 × 792. In PyMuPDF the origin
(0, 0) is the **top-left** corner and y grows **downwards**. A box is
`(x_min, y_min, x_max, y_max)`.

When you render a page at *dpi* dots per inch, `pixel = point × dpi / 72`. At 150 dpi,
the point (100, 200) becomes the pixel (208, 417). You will use this conversion all week.

**What N3 must produce:** records with the *same fields* as `FAKE_PAPER` in
[stub_paper.py](src/gea/stub_paper.py), so the nodes do not care which backend made them.

```text
chunks : id, page, section, bbox, text           (paragraph text, ~150 words each)
tables : id, label ("Table 2"), page, bbox, caption
figures: id, label ("Figure 3"), page, bbox, caption
pages  : page, size (w, h in points), image path (150 dpi PNG)
```

**Why keep coordinates?** Because on Day 6 every claim must be boxed on the page. Text
without positions can answer a question, but it cannot prove the answer.

### Steps

**2.1 🔌 Choose the questions:** `scripts/select_questions.py` (1 h). It reads the local
`data/spiqa/test-A/SPIQA_testA.json` (downloaded yesterday):
- Group questions by paper. Shuffle the papers with `random.Random(42)`.
- The first ~5 papers → **dev** (keep 20 questions). The following papers → **test**, until
  you have 150 questions (about 30 papers).
- Turn each reference into a label with `re.search(r"(Figure|Table)(\d+)", name)`, giving
  "Table 2", and take the kind from `all_figures[name]["content_type"]`.
- Save one JSON line per question to `eval/splits/dev.jsonl` and `eval/splits/test.jsonl`,
  with fields `qid, paper_id, question, answer, reference, ref_label, ref_kind`. Use
  `atomic_write_text` from safeio.
- Print how many references are tables and how many are figures. Aim for a rough balance.

*Why the splits go in `eval/` (in git), not `data/`:* they define the experiment. Anyone
must be able to rerun exactly the same questions.

**2.2 🔌 Write real N3:** `src/gea/ingest.py` (3–4 h). One function,
`ingest_pdf(pdf_path, out_dir) -> dict`:

1. **Pages:** render each page at 150 dpi and save it with
   `with atomic_path(out_dir / f"page_{n}.png") as tmp: pix.save(tmp)`; keep
   `page.rect.width/height` as the page size.
2. **Text blocks:** `page.get_text("dict")["blocks"]` (type 0 = text). Join a block's spans
   into its text and keep the block's `bbox`.
3. **Captions:** blocks matching `^(Figure|Fig\.|Table)\s*(\d+)\s*[:.]`.
4. **Table regions:** `page.find_tables()`. Give each table the closest "Table N" caption
   (usually just above it).
5. **Figure regions:** join `page.get_image_info()` boxes and `page.cluster_drawings()`
   boxes (vector plots) that lie just above a "Figure N" caption.
6. **Chunks:** text blocks that are not captions and not inside a table or figure region.
   Merge neighbours in reading order into ~150-word chunks; the chunk box is the union of
   its block boxes.
7. **Sections:** short blocks matching `^\d+(\.\d+)*\s+[A-Z]` are headings; each chunk
   remembers the last heading's number.

> **What real papers needed (built on Day 2, in [ingest.py](src/gea/ingest.py)).** Looking
> at real pages showed that the simple version above fails in several ways, each now fixed
> and kept fixed by [tests/test_ingest.py](tests/test_ingest.py):
>
> - Captions are found **per row**, not per block: PyMuPDF glues a caption to the table rows
>   under it. A caption stops at the first row whose pieces are more than one font size
>   apart (a table row). IEEE captions ("TABLE II" alone, title on the next line) count too.
> - `find_tables()` misses many tables and splits others, so a guess is used only right next
>   to a table caption, then **grown** over table text beside it (a missed column, header).
> - A paper puts its tables either under or over their captions; the reader **counts which**
>   over the whole paper (two passes) and searches that side first.
> - Table rows that look like text (body font, several lines) are recognised as
>   **table-like** (mostly gapped rows or mostly numbers), so they are not paragraphs.
> - Page numbers and running headers (the top and bottom 40 points) are ignored; a region
>   under 1% of the page, or found only by the last-resort search, is marked `doubtful`.
> - Each paragraph keeps the section it is in; text before the first heading is `front`;
>   words broken at a line end are joined ("be-" + "tween").

Save with `atomic_write_json(out_dir / "corpus.json", ...)` **as the very last step**.
`scripts/build_corpus.py` loops over the chosen papers and **skips any paper whose
`corpus.json` already exists**.
*Why `corpus.json` comes last:* it is the "this paper is finished" mark. If the power dies
halfway through a paper, there is no `corpus.json`, so the next run does that paper again
from the start. Its leftover page images are simply overwritten.
*Why regions come from captions:* every real table and figure has a numbered caption, and
the number is how you match SPIQA's reference ("Table 2") to your record.

**2.3 🔌 Coverage report** (1 h). For every selected question: is there a record whose label
equals `ref_label` on some page? Print the percentage and the misses. For the misses, draw
the detected regions on the page image (PIL `ImageDraw.rectangle`), save the drawings to
`data/debug/`, and copy them to your laptop or phone.
**🔋 Look at them during the next outage**: why was the region missed? Fix the biggest cause
in the next power window. If a few remain, drop those questions and **write down how many and
why**; the thesis must report it.

> **Result:** [scripts/build_corpus.py](scripts/build_corpus.py) reads all 36 papers (475
> pages, about 90 seconds on the CPU) and reports **100% coverage: 20/20 dev and 150/150
> test**, with no question dropped. Of all 364 tables and figures, 3 are marked `doubtful`.
> The numbers are in `results/day2_coverage.json`.

**2.4 🔌 Connect it to the graph** (30 min). In [corpus.py](src/gea/corpus.py), make
`get_corpus` load `data/corpus/<paper_id>/corpus.json` when `cfg.BACKEND == "real"`.
(The indexes are added tomorrow.)

**2.5 🔌 Notebook section "Day 2: real ingestion"**: show one real paper's page with its
chunks, tables and figures drawn as coloured boxes. That picture goes in your thesis.

### Check it works

- [ ] every record box lies inside its page (reuse `on_page` from notebook test 5A, with the real page size)
- [ ] coverage ≥ 85 % of references found by label
- [ ] the debug image of 3 random pages looks right to your eye
- [ ] **power-cut test:** stop `build_corpus.py` with Ctrl+C halfway, run it again: it continues and the result is the same
- [ ] `pytest -q` still green (the stub backend is untouched)

### Write 🔋 (45 min)

Implementation → *Data and Ingestion*: which SPIQA split, how papers and questions were
chosen, how regions are found, the coverage number, and what was dropped.

### Explain it back 🔋

<details><summary>1. A figure box is (100, 200, 300, 400) in points. Where is it in a 150 dpi image?</summary>

Multiply by 150/72 ≈ 2.083: (208, 417, 625, 833) pixels.
</details>

<details><summary>2. Why must you never tune a prompt on the test questions?</summary>

The test result is then partly memory of the test, not quality of the system, so it
overestimates how well it works on new questions. Dev is for tuning; test is touched once.
</details>

<details><summary>3. Why is `corpus.json` written last, after all the page images?</summary>

It marks the paper as finished. If it existed before the images were done, a power cut would
leave a paper that looks finished but has missing images, and the next run would skip it.
</details>

### If you're behind

Skip `cluster_drawings()` and use only images plus captions; vector figures will then be
found tomorrow by ColQwen2's similarity map instead. Use 25 test papers instead of 30.
**Never skip** the coverage report.

---

## Day 3: The two indexes (N4) and text search (N6a)

**Power plan.** 🔌 about 5 h, including ~30 min of GPU indexing (resumable per paper). 🔋:
the embeddings reading (do the toy examples with pen and paper); the Write section.

**Goal:** every paper has a text index (SPECTER2 + FAISS) and a page index (ColQwen2), and
N6a really searches.

**By tonight:** `data/corpus/<id>/text.faiss`, `pages.pt` · `src/gea/indexes.py` · real
N6a · a first retrieval number for RQ2.

### Understand it first 🔋

**Embeddings: meaning as coordinates.** An embedding model turns a text into a list of
numbers, a point in space, so that **similar meaning ends up close together**. A toy example
in 2 dimensions:

| text | vector |
|---|---|
| "cat" | (0.9, 0.1) |
| "kitten" | (0.85, 0.2) |
| "car" | (0.1, 0.95) |

"cat" and "kitten" point the same way; "car" points elsewhere. Real embeddings have 768
numbers instead of 2, but the idea is identical.

**Cosine similarity** measures how much two vectors point the same way: 1 = same direction,
0 = unrelated. If every vector is first scaled to length 1 ("normalised"), cosine similarity
is just the dot product: multiply number by number and add up.
*Pen and paper:* cat · kitten = 0.9×0.85 + 0.1×0.2 = 0.785; cat · car = 0.09 + 0.095 = 0.185.

**SPECTER2.** A text embedder trained on millions of scientific papers, so it knows that
"low-resource" and "few labelled examples" are related. It has small add-on layers called
**adapters**. One adapter is for documents (`proximity`) and one is for short search
questions (`adhoc_query`). Questions and paragraphs are written very differently, so each
side gets the adapter made for it, like one translator wearing two different hats.

**FAISS.** A library that finds the nearest vectors quickly. With a few thousand chunks,
the simple `IndexFlatIP` (compare with every vector, IP = inner product) is exact and
instant. Fancier indexes only matter at millions of vectors.

**ColPali / ColQwen2: searching pages as pictures.** Instead of reading a page's text
(which breaks on tables, formulas and plots), ColQwen2 looks at the page **image**, cuts it
into small squares (**patches**) and gives *each patch* its own vector. A question gets one
vector *per word*. The page score is **MaxSim**: for each question word, find its
best-matching patch, then add those best scores up.

A toy example. Question "low-resource accuracy"; a page with 4 patches:

| | P1 title | P2 table row "Low-resource 71.3" | P3 chart | P4 text |
|---|---|---|---|---|
| low-resource | 0.1 | **0.9** | 0.2 | 0.3 |
| accuracy | 0.2 | **0.7** | 0.6 | 0.4 |

Page score = 0.9 + 0.7 = **1.6**. And look: the high numbers sit on **P2**. That column is
the **similarity map**, which tomorrow tells you *where* on the page the answer is.

*Why many vectors instead of one ("late interaction"):* squeezing a whole page into one
vector blurs the details. Keeping one vector per patch keeps the small table cell
findable.

**Why two indexes?** Text search is best for claims written in sentences; page-image search
is best for tables and figures. Both point to the same page numbers, so a hit in one leads
to the same place in the other.

### Steps

**3.1 🔌 Stop llama-server** (Ctrl+C in its window). You need the GPU memory for ColQwen2.

**3.2 🔌 SPECTER2 encoder:** [src/gea/indexes.py](src/gea/indexes.py) (1 h). Runs on the CPU.

```python
@lru_cache(maxsize=1)
def specter():                                              # loaded once per session (~4 s)
    base = local_snapshot("allenai/specter2_base")          # the downloaded folder, never the internet
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoAdapterModel.from_pretrained(base)
    model.load_adapter(local_snapshot("allenai/specter2"), load_as="proximity")
    model.load_adapter(local_snapshot("allenai/specter2_adhoc_query"), load_as="adhoc_query")
    return tok, model.eval()

@torch.inference_mode()
def embed_text(texts: list, kind: str = "doc"):
    tok, model = specter()
    model.set_active_adapters("proximity" if kind == "doc" else "adhoc_query")
    out = []
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], padding=True, truncation=True, max_length=512, return_tensors="pt")
        cls = model(**enc).last_hidden_state[:, 0]          # the first token's vector = the text's embedding
        out.append(torch.nn.functional.normalize(cls, dim=-1))
    return torch.cat(out).numpy().astype("float32")         # (n, 768), length 1 each
```

> **Built on Day 3.** The model loads from the local download only (`local_files_only`): a
> power cut often takes the internet with it, and a missing model should fail at once, not
> hang. It stays on the CPU (110M weights, about 4 chunks a second while another job uses the
> machine). [tests/test_indexes.py](tests/test_indexes.py) checks length-1 vectors, that
> batching and padding change nothing, that the two adapters differ, and the 512-token cut.
>
> **What it does and does not understand** (real scores, question: "How well does the method
> work when only a few labelled examples are available?"):
>
> | paragraph | shared words | SPECTER2 |
> |---|---|---|
> | A answers it ("low-resource setting … 71.3% accuracy") | none | 0.769 |
> | B only shares words ("labelled examples were collected by three annotators") | 2 | **0.779** |
> | C unrelated ("20 epochs on four GPUs") | none | 0.733 |
>
> Over four such questions the unrelated paragraph was always last, but the answer beat the
> word-sharer only twice. **SPECTER2 matches topics, not answers**: it was trained to match
> a search with papers on the same topic. That is why the agent keeps the top 3, asks the
> page index for a second opinion, and lets G1 check the evidence. Step 3.6 measures it on
> real questions.

**3.3 🔌 Text index per paper** (30 min). Embed every chunk **and** every caption with
`kind="doc"`. Then `faiss.IndexFlatIP(768)` → `add` → save with
`with atomic_path(folder / "text.faiss") as tmp: faiss.write_index(index, str(tmp))`. Save
the matching record ids in the same order. Also keep the IDF keyword table from the stub N4:
the stub G1 and N9 still use `score()` until Days 5–6. Skip papers whose `text.faiss` already
exists.

> **Built on Day 3:** `python scripts/build_indexes.py text` (CPU, ~10 min) or `... text --gpu`
> (B580, **0.9 min** for 2,532 texts from 36 papers). Each paper gets `text.faiss` plus
> `text_ids.json` (which record each row is: chunk `c3`, or the caption of `t1`/`f2`). The ids
> are written first and the index last, so an index that exists is complete; an index older
> than its `corpus.json` is refused. Before loading a model the script checks free memory and
> the GPU memory other programs hold, and refuses if they are short: running out crashes
> *every* program on the PC, including another experiment.

**3.4 🔌 Page index per paper** (1–2 h, mostly waiting; resumable). On the GPU:

```python
from colpali_engine.models import ColQwen2, ColQwen2Processor

model = ColQwen2.from_pretrained("vidore/colqwen2-v1.0", torch_dtype=torch.bfloat16).to("xpu").eval()
proc = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

@torch.inference_mode()
def embed_page(image):                                     # one PIL page image
    batch = proc.process_images([image]).to(model.device)
    emb = model(**batch)[0]                                # (tokens, 128)
    mask = batch["input_ids"][0] == proc.image_token_id    # which tokens are image patches
    return emb[mask].float().cpu()                         # (n_patches, 128)
```

Also store the patch grid size (rows × columns) for every page:
`proc.get_n_patches(image.size, patch_size=model.patch_size, spatial_merge_size=model.spatial_merge_size)`.
For each paper, save everything with `with atomic_path(folder / "pages.pt") as tmp:
torch.save(pages, tmp)`, and **skip papers whose `pages.pt` exists**. A power cut costs at
most the one paper in progress (about 30 seconds of work).
Do one page per call: it is simpler, and nothing gets padded.

> **Check the names** in your installed colpali-engine (`help(ColQwen2Processor)`):
> `image_token_id` and `get_n_patches` have moved between versions. The idea stays the same:
> keep only the image-patch vectors, and remember the grid shape.

Then **unload**: `del model; empty_cache(DEVICE)`.

> **Built on Day 3:** `python scripts/build_indexes.py pages`: **475 pages in 3.0 minutes** on
> the B580 (about 0.3 s a page; the 1–2 h above was far too cautious). The model is
> `colqwen2-base` with the `v1.0` add-on merged in, loaded from the local download only. A
> 150-dpi page (1275×1650 px) is shrunk to 672×868 and becomes **31 × 24 = 744 patch
> vectors**. Each patch is about 25.5 points (9 mm) square, stored row by row, so patch `k` sits
> at row `k // 24`, column `k % 24`. About 1.6 MB per paper.

**3.5 🔌 Real N6a** (1 h). In [retrieval.py](src/gea/retrieval.py), at the top of
`text_retriever`:

```python
if cfg.BACKEND == "real":
    return real_text_retriever(corpus, query, section, k)
```

`real_text_retriever` embeds the query with `kind="query"`, searches the FAISS index, keeps
chunks from N2's section when one was named (otherwise all chunks), and returns the same
`Evidence` objects as the stub. **The node body does not change.**

> **Built on Day 3.** `real_text_retriever` returns the top **3** paragraphs
> (`cfg.TEXT_TOP_K`; caption rows are left for the table and figure tools). "Section 4" keeps
> 4, 4.1, 4.2… and falls back to the whole paper when nothing matches. The ranking is chosen by
> `cfg.TEXT_SEARCH`, and `python scripts/compare_text_search.py` measured all three on dev:
>
> | ranking | a correct paragraph 1st | in the top 3 |
> |---|---|---|
> | keyword | 45% | **70%** |
> | SPECTER2 | 15% | 35% |
> | hybrid (both, by rank fusion) | **50%** | 60% |
>
> Keyword search wins because SPIQA questions reuse the paper's own words ("GRID", "ripple
> sets"). **Chosen: hybrid.** It is best at first place, 2 of 20 behind keyword at top 3, and
> it keeps a meaning signal for the planner's reworded sub-queries (Day 5). Re-check it then.

**3.6 🔌 First retrieval numbers** (1 h). On the 20 dev questions:
- **Page hit@k:** load ColQwen2 *on the CPU* (`.to("cpu")`, bf16), embed the question, score
  every page of the paper with MaxSim, and check whether the reference's page is in the top
  1 / top 3.
- **Text hit@3:** whether one of the top-3 chunks is on the reference's page or mentions its
  label.

```python
def maxsim(q, pages):                       # q: (q_tokens, 128); pages: list of (n_patches, 128)
    return [float((q @ p.T).max(dim=1).values.sum()) for p in pages]
```

Save the numbers with `atomic_write_json` to `results/day3_retrieval.json`. They are early
evidence for **RQ2** ("can visual embeddings find the right tables and figures?").

> **Result (Day 3):** `python scripts/eval_retrieval.py` (20 s on the B580) →
> `results/day3_retrieval.json`:
>
> | | measured | random guess |
> |---|---|---|
> | page hit@1 (ColQwen2) | **75%** | 12% |
> | page hit@3 (ColQwen2) | **95%** | 37% |
> | text hit@3 (SPECTER2), strict: caption or label named | **50%** | |
> | text hit@3, lenient: also "on the same page" | 75% | |
>
> Only one question's page fell outside the top 3 (1708.00160v2, Figure 2: rank 10 of 15).
> First evidence for RQ2: looking at page *images* finds the right page far better than chance.

**3.7 🔌 Notebook section "Day 3: the indexes"**: one dev question, its top-3 pages with
scores, and its top-3 chunks.

> **Built on Day 3: Section 18.** 18A shows what is inside one paper's two indexes. 18B is the
> page search for "Which dataset has the most 4-hop triples?": page 6 first, with Table 1 boxed,
> using the saved scores so ColQwen2 is not loaded. 18C runs the hybrid text search live: the
> paragraph naming Table 1 comes 3rd. 18D shows Day 3's numbers. The point: both searches find
> *where* the answer is, but the answer is a table cell, which is Day 4's job.

### Check it works

- [ ] a chunk searched with its own text comes back first, with similarity ≈ 1.0
- [ ] `pages.pt` exists for every paper; every page's grid rows × columns = its number of patch vectors
- [ ] page hit@3 on dev measured and saved
- [ ] **power-cut test:** stop the page indexing with Ctrl+C, restart it: it skips the finished papers
- [ ] `pytest -q` still green

### Write 🔋 (45 min)

Implementation → *Indexing*: SPECTER2 with two adapters, FAISS flat inner product,
ColQwen2 page embeddings, MaxSim. Include the toy MaxSim table; examiners like it.

### Explain it back 🔋

<details><summary>1. Why does SPECTER2 use a different adapter for the question than for the paragraphs?</summary>

Questions are short and phrased as questions; paragraphs are long statements. Training a
separate small add-on for each side ("asymmetric search") places a question near the
paragraph that *answers* it, not near paragraphs that merely look similar.
</details>

<details><summary>2. In MaxSim, why take the max over patches and then the sum over words?</summary>

The max finds, for each word, the one place on the page that matches it best. The sum makes
a page score high only if *every* word found a good match somewhere. That rewards pages
that contain all parts of the question.
</details>

<details><summary>3. Why is ColQwen2 on the CPU at question time but on the GPU today?</summary>

Today it encodes hundreds of page *images*, which is heavy and needs the GPU. At question
time it encodes one short *text* query, which is light, and the GPU must stay free for the
language model (7 GB) because both do not fit in 12 GB.
</details>

### If you're behind

Skip the section filter in 3.5. **Never skip** 3.6; it is your first RQ2 evidence.

---

## Day 4: Tables, figures and zoom (N6b, N6c, N7)

**Power plan.** 🔌 about 6 h of coding and testing. 🔋: the reading; checking the drawn table
grids and heat maps (copied to the laptop or phone); the Write section.

**Goal:** N6b returns a real table with every cell's box and text, N6c returns a real figure
with its similarity map, and N7 returns a sharp crop.

**By tonight:** real N6b, N6c, N7 · on 5 dev questions the right table or figure is found and
drawn.

### Understand it first 🔋

**The similarity map → a box.** Yesterday's table had one column per patch. Take, for each
patch, the best score over the question's words: one number per patch. Arranged in the
page's grid, that is a **heat map** of the page. `patches_to_bbox` (already in
[proofs.py](src/gea/proofs.py)) keeps the hot patches (≥ 50 % of the hottest), draws the
smallest rectangle around them, and converts patch positions into points:
`x = column × page_width / n_columns`.

**Choosing the region on a page.** ColQwen2 says *which page*. Your Day 2 records say where
the tables and figures *are* on that page. Score each region by how hot the map is inside
it (the mean of its best patches), then pick the hottest region of the kind you need (a
table for N6b, a figure for N6c).

**Table structure recognition.** The **Table Transformer** is an object detector: it looks
at a table image and draws boxes around every **row** and every **column**. A **cell** is
where a row box and a column box cross. You then read each cell's text from the PDF's own
text layer at that position (`page.get_textbox(rect)`). No OCR is needed, so no OCR errors,
and every cell keeps its exact box. That box is what N10b will draw.

**Why zoom (N7)?** The language model sees an image at limited resolution. A figure 300
points wide rendered at 150 dpi is 625 pixels across, so small axis labels blur.
Re-rendering only that region at 300 dpi gives 1250 pixels: it is a magnifying glass. The
router zooms when a figure covers less than 30 % of its page (`cfg.ZOOM_IF_AREA_BELOW`).

### Steps

**4.1 🔌 Query-time ColQwen2 on the CPU** (1 h). In `indexes.py`: a cached
`colqwen_cpu()` loader (bf16, CPU) and `embed_query(text) -> (q_tokens, 128)`.
`rank_pages(corpus, query)` returns the pages sorted by MaxSim plus each page's
similarity map:

```python
def similarity_map(q, page_vectors, rows, cols):
    sim = (q @ page_vectors.T).max(dim=0).values       # best word score for each patch
    return sim.view(rows, cols).numpy()                 # patch vectors are stored row by row
```

**4.2 🔌 Test the map's orientation** (30 min). This is the classic bug: rows and columns
swapped. Take 3 dev figures you know, compute `patches_to_bbox` on their map, and compute
the IoU with the figure region from Day 2. If it is near 0, try `.view(cols, rows).T`.
Write this as a test.
**Also:** `patches_to_bbox` uses `cfg.PAGE_W/H` (the fake A4 page). Give it the real page
size as an argument; real papers are often US Letter (612 × 792).

**4.3 🔌 Table structure** (2 h). In `indexes.py`:

```python
from transformers import AutoImageProcessor, TableTransformerForObjectDetection
NAME = "microsoft/table-structure-recognition-v1.1-all"      # downloaded on Day 1
tatr_proc = AutoImageProcessor.from_pretrained(NAME)
tatr = TableTransformerForObjectDetection.from_pretrained(NAME).eval()

def table_cells(page, region, dpi=150):
    pix = page.get_pixmap(dpi=dpi, clip=pymupdf.Rect(region))
    img = PILImage.frombytes("RGB", (pix.width, pix.height), pix.samples)
    with torch.inference_mode():
        out = tatr(**tatr_proc(images=img, return_tensors="pt"))
    det = tatr_proc.post_process_object_detection(out, threshold=0.6, target_sizes=[img.size[::-1]])[0]
    to_pt = lambda b: (region[0] + b[0] * 72 / dpi, region[1] + b[1] * 72 / dpi,
                       region[0] + b[2] * 72 / dpi, region[1] + b[3] * 72 / dpi)
    found = [(tatr.config.id2label[int(l)], to_pt(b)) for l, b in zip(det["labels"], det["boxes"].tolist())]
    rows = sorted((b for name, b in found if name == "table row"), key=lambda b: b[1])
    cols = sorted((b for name, b in found if name == "table column"), key=lambda b: b[0])
    grid = [[(c[0], r[1], c[2], r[3]) for c in cols] for r in rows]       # cell = column's x-range x row's y-range
    text = [[page.get_textbox(pymupdf.Rect(b)).strip() for b in row] for row in grid]
    return grid, text
```

The first row is the header and the first column is the row's name. Build exactly the stub's
format: `header`, `rows`, `cell_boxes`. Then `table_text()` and the stub's `cells` list
work unchanged. **Cache** every parsed table with `atomic_write_json` to
`data/corpus/<id>/tables/<table_id>.json`: the same table is often retrieved again, and the
cache survives power cuts.
**Fallback:** if TATR finds fewer than 2 rows or 2 columns, try
`page.find_tables(clip=region)` and use its cells.

**4.4 🔌 Real N6b and N6c** (2 h). Same pattern as N6a: `if cfg.BACKEND == "real": return
real_table_retriever(...)`. Evidence fields:
- table: `kind="table"`, `label`, `content=table_text(...)`, `cells`, `bbox`
- figure: `kind="figure"`, `label`, `content=caption`, `bbox`, and **`similarity=` the page's
  map** (N10b needs it)

**4.5 🔌 Real N7** (45 min). Re-render the region at 300 dpi: `page.get_pixmap(dpi=300,
clip=rect)`. Shrink it so the long side is at most 1024 px (images cost model tokens),
then store the PNG bytes in `Evidence.crop`. `route_zoom` must use the *real* page size,
not `cfg.PAGE_W/H`.

**4.6 🔌 Smoke test** (1 h). Run the **whole graph** with `cfg.BACKEND = "real"` and the
**rules** planner on 5 dev questions. The stub G1 and N9 still work, because the real corpus
keeps the IDF table. The answers will be clumsy; the point is that **real evidence flows
through every node** without crashing.

**4.7 🔌 Notebook section "Day 4: tables and figures"**: one table with its cell grid drawn, one
figure with its heat map laid over the page. Save both images to `data/debug/` and check
them 🔋 during the next outage.

> **Built on Day 4.**
>
> - **Tables.** [src/gea/tables.py](src/gea/tables.py) parses every table once:
>   `python scripts/build_indexes.py tables`, 159 tables in 0.9 min on the CPU, 157 by the
>   Table Transformer. A word belongs to the cell that holds its *centre*, because touching
>   a box would take the next row's words.
> - **Choosing a table or figure.** By its *words* (the caption plus the text printed inside
>   it) and by ColQwen2's score for its **page**. On dev, the right one comes 1st for **85%** of
>   questions (random 26%): words alone 80%, page score alone 80%. Captions alone gave tables
>   only 6/9.
> - **The heat map is noisy.** It lights up the question's words wherever they appear (the
>   figure, its caption, the paragraph about it), so its hottest patch lands on the reference
>   only 15% of the time (random 8%). Choosing by the heat inside a region scored 55%. So the
>   heat only narrows the box *inside* the chosen region (median IoU 0.81 with the region).
>   The orientation was checked: the swapped grid is half as hot on the reference.
> - **When ColQwen2 does not fit** (another job holds the GPU or memory), query vectors come
>   from `data/cache/page_queries/` or are skipped. The words then decide, and a figure's box
>   is its N3 region.
> - **Smoke test.** `python scripts/smoke_real.py` ran 5 dev questions through the whole
>   graph without errors. Notebook Section 19 shows the work.

### Check it works

- [ ] for 3 dev tables, the text in each cell box equals what you see in the PDF
- [ ] figure box vs Day 2 region: IoU ≥ 0.5 on most dev figures
- [ ] 5 dev questions run end to end on `BACKEND="real"`
- [ ] `pytest -q` still green

### Write 🔋 (45 min)

Implementation → *Retrieval tools*: page ranking, region choice, table structure, the
zoom, and the fallback. Include the heat-map image.

### Explain it back 🔋

<details><summary>1. Why read cell text from the PDF instead of with OCR?</summary>

A digital PDF already contains the exact characters at known positions. Reading them
there is exact, while OCR can misread "71.3" as "71.8". The Table Transformer only supplies
*where* the cells are.
</details>

<details><summary>2. What breaks if the similarity map's rows and columns are swapped?</summary>

Every figure box lands in the wrong place, often a mirrored region of the page, so CGS
collapses even though retrieval looks fine. That is why 4.2 tests it against known figures.
</details>

### If you're behind

Use only `find_tables()` for cells and skip TATR. Say so in the thesis, as "a rule-based
table parser". **Never skip** 4.2.

---

## Day 5: The model wakes up (N1, N2, N5, G1)

**Power plan.** 🔌 about 5 h with llama-server running. 🔋: the reading; **drafting all four
prompts on paper** (N1, N2, G1, plus the N5 prompt you already have) so the power time is
spent testing them, not inventing them; the prompt changelog.

**Goal:** the four "thinking" nodes use Qwen2.5-VL through llama.cpp, always answer in valid
JSON, and their answers are cached to disk.

**By tonight:** `call_llm()` with a power-cut-safe cache · real N1, N2, N5, G1 · 10 dev
questions traced and read by you.

### Understand it first 🔋

**A language model as a function.** Text (and images) in, text out. llama-server makes your
local model look like the OpenAI web API: you send JSON to
`http://localhost:8080/v1/chat/completions` and get JSON back. That is all `llm.py` does.

**Quantisation (4-bit).** A 7B model has 7 billion numbers (weights). Stored with 16 bits
each, that is about 14 GB, too much for 12 GB. Stored with about 4 bits, it is 4.4 GB.
It is like saving a photo as a smaller JPEG: slightly less exact, much smaller. Mention the
small quality cost in your limitations.

**JSON schema = forced answer shape.** You give llama.cpp a schema such as "an object with
`tool` ∈ {text, table, figure} and `query`: string". llama.cpp then only allows tokens
that keep the answer valid. The model *cannot* invent a tool or write broken JSON. This
turns "parse the model's prose and hope" into "read a field".

**Temperature 0 + seed + cache = reproducible, and power-cut-proof.** Temperature 0 means
"always pick the most likely next token", so the same input gives the same output. The
**cache** stores every answer on disk under a fingerprint (hash) of the exact request, in
`results/llm_cache/`, which is in git, so every checkpoint also backs it up.
Asking the same thing again reads the answer from disk: instant, free and identical. Under
load shedding this matters twice: **after a power cut, every model call already made comes
back from the cache in milliseconds**, so a re-run quickly catches up to where it stopped.

**"The model proposes, the rules decide."** Already in N5: the model *suggests* a tool, and
the hard rules (the tool must exist; never repeat a sub-goal + tool pair; at most 3 rounds)
always apply. A clever model can make the system better, but it can never make it loop
forever or call a tool that does not exist.

**LLM-as-judge for G1, and its two failure modes.** A **too-lenient** judge says "enough"
too early, which leads to hallucination. A **too-strict** judge never says "enough", which
leads to abstaining too often. So G1 gets a **rule floor**: the model must say "sufficient"
**and** every table or figure sub-goal must have at least one item of that kind in the
buffer. You measure both failure modes on Day 8 (wrong answers vs abstentions).

### Steps

**5.1 🔌 Start the server** in its own terminal, and make it a habit after every power cut:

```bash
bash scripts/start_llm.sh
```

**5.2 🔌 `call_llm` with a cache** (1.5 h). In [llm.py](src/gea/llm.py):

```python
from gea.safeio import atomic_write_json

PROMPT_VERSION = "v1"        # raise it whenever you edit a prompt -> old cache entries stop matching

def call_llm(messages, schema, name, images=()):
    if images:                                     # PNG bytes -> attached to the last user message
        parts = [{"type": "text", "text": messages[-1]["content"]}]
        parts += [{"type": "image_url", "image_url": {"url": "data:image/png;base64," +
                   base64.b64encode(png).decode()}} for png in images]
        messages = messages[:-1] + [{"role": "user", "content": parts}]
    body = {"model": cfg.LLM_MODEL, "temperature": 0, "seed": cfg.SEED, "messages": messages,
            "response_format": {"type": "json_schema", "json_schema": {"name": name, "schema": schema}}}
    key = hashlib.sha256(json.dumps([PROMPT_VERSION, body], sort_keys=True).encode()).hexdigest()
    path = cfg.LLM_CACHE_DIR / key[:2] / f"{key}.json"            # results/llm_cache, in git
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    ...  # POST to f"{cfg.LLM_URL}/chat/completions" exactly like llm_plan, json.loads the content
    atomic_write_json(path, answer)                # never a half-written cache entry
    return answer
```

Rewrite `llm_plan` to use it. (The `key[:2]` sub-folder keeps any one folder from holding
thousands of files.)

**5.3 🔌 Real N1** (1 h). Prompt: *"Split the question into at most 3 sub-questions. Each needs
ONE kind of evidence: text (paragraphs), table, or figure. A simple question stays one
sub-question."* Schema: `sub_goals`, an array of 1–3 items `{text, modality ∈ MODALITIES}`.
If the call fails, fall back to the regex version and record that it happened (the same
pattern as N5).
*Why a model here:* SPIQA questions rarely say "table". *"Which method has the lowest
error?"* needs a table, and only a model that reads meaning knows that.

**5.4 🔌 Real N2** (45 min). `query_embedding` = `embed_text([q], "query")[0]`, a real 768-number
vector. The filters (section / table / figure numbers, entity names) come from the model with
a small schema, with the regex as fallback.

**5.5 🔌 Real N5** (15 min). Set `cfg.PLANNER_MODE = "llm"` for real runs. Section 14 already
built and tested this.

**5.6 🔌 Real G1** (1.5 h). Show the model the question, the sub-goals and every buffer item as
text: the paragraph, the `table_text`, the figure caption plus *"(image attached)"*. Attach
crops only for figure sub-goals. Schema:
`{sufficient: bool, uncovered: [int], missing: string}`. Then apply the rule floor described
above.

**5.7 🔌 Read 10 dev traces** (2 h). Run with `cfg.TRACE = True` and *read every line*. Copy the
traces into `notes/traces_day5.md` so you can also read them 🔋 during an outage. Keep a
**prompt changelog** in `notes/prompts.md`: what you changed and why. Raise `PROMPT_VERSION`
after each change.

### Check it works

- [ ] 10 dev questions: every model reply is valid JSON (the schema guarantees it; check anyway)
- [ ] run the same question twice: the second run is instant and gives identical output (the cache)
- [ ] **power-cut test:** stop llama-server mid-run, restart it with `start_llm.sh`, re-run: the finished calls come from the cache
- [ ] G1 says INSUFFICIENT for a table question when only text is in the buffer
- [ ] with the server stopped, the nodes fall back to the rules, and the trace says so
- [ ] `pytest -q` still green (the tests use `PLANNER_MODE="rules"`)

### Write 🔋 (45 min)

Implementation → *The language model*: the model, 4-bit quantisation, llama.cpp, JSON
schemas, temperature 0, the cache, "model proposes, rules decide", G1's rule floor. Put the
prompts in an appendix.

### Explain it back 🔋

<details><summary>1. Why can't the planner crash the system by choosing a tool that doesn't exist?</summary>

Two walls. The JSON schema only allows the three tool names, and N5's rules overrule an
invalid or repeated choice anyway. The loop bound holds whatever the model says.
</details>

<details><summary>2. What does the cache make possible?</summary>

Re-running everything (for a new metric, a fixed bug in scoring, an examiner's request, or
simply after a power cut) gives exactly the same answers without hours of GPU time. It is
also the proof that the numbers in the thesis came from these exact model replies.
</details>

<details><summary>3. Why give G1 a rule floor instead of trusting the model?</summary>

A judge model can be too lenient, saying "enough" when the table was never retrieved, and
that is exactly the failure that produces hallucinations. The floor makes the minimum
requirement (evidence of the right kind exists) a fact checked in code.
</details>

### If you're behind

Keep N2's filters as regex (only its embedding becomes real). **Never skip** the cache.

---

## Day 6: Writing and checking the answer (N9, G2, N10)

**Power plan.** 🔌 about 6 h with llama-server running. 🔋: the reading; drafting the N9 and
support-check prompts on paper; **looking at the proof images** (copied to the laptop or
phone) and noting which boxes are wrong.

**Goal:** the model writes cited claims from the buffer, G2 checks them with equality tests
**plus** a model support check, and N10b draws real boxes.

**By tonight:** real N9, G2, N10b · a real answer with a proof image, made with
`draw_proof` on a real PDF.

### Understand it first 🔋

**Grounded generation.** N9 receives only the buffer items, each labelled E1, E2, …, with
the table text and image crops. Its instruction: *answer using only these items, and attach
to every claim the item it rests on*. It never sees the paper itself.

**Three kinds of citation:**

| Kind | Looks like | Can be checked by |
|---|---|---|
| quote | `{"evidence": "E1", "quote": "…exact sentence…"}` | code: is the text in E1? |
| cell | `{"evidence": "E2", "cell": ["Low-resource", "Gain"], "value": "+1.7"}` | code: does that cell say +1.7? |
| observation | `{"evidence": "E3", "observation": "the loss flattens after epoch 10"}` | only a reader of the image, i.e. the model |

**G2 in two layers.**
1. **Equality tests** (already built): the evidence exists, the quote is word for word, the
   cell value matches, and every number in the sentence appears in a citation. These are
   exact, cheap and impossible to argue with.
2. **Support check** (new, one model call for all claims): *"Does this evidence support
   this claim **as written**, including which row or subject it is about?"*

**Why layer 2 is needed.** Notebook Section 15 found the boundary: a claim can cite a *real*
cell from the *wrong row* ("the low-resource gain is +4.4", citing High-resource/Gain), and
every equality test passes. Figure observations cannot be equality-checked either. The
support check reads meaning, so it can catch both. Whether it actually does is a
**result**: test it in 6.3.

**Faithful is not the same as correct.** G2 checks that the answer is **faithful** to the
evidence. If the buffer holds the wrong table, a faithful answer can still be wrong. That
is why Day 8 measures retrieval (RF) separately from correctness.

**Why N10a is "claim links" now (decision 2).** G2 has already matched every claim to its
exact source, so the link is *known*, not estimated. Integrated gradients would estimate
which input words influenced the output, and would need gradients through a full-precision
7B model (about 14 GB, more than the B580 has). Write this down as a design decision with
its reason; examiners respect a justified change.

### Steps

**6.1 🔌 Real N9** (2.5 h). Prompt: the question + every buffer item as `[E1] (text, Section 4)
…`, `[E2] (table, Table 3) caption + rows`, `[E3] (figure, Figure 2) caption`, with crops
attached. Schema: `claims`, an array of `{text, cites, conclusion}`, where each cite is one of
the three shapes, combined with `anyOf`.
**Trick:** build the schema per call with `"evidence": {"enum": ["E1", "E2", …]}`, so the
model *cannot* cite an item that is not in the buffer.
**On a rewrite:** add G2's reasons ("claim 2: E2 Low-resource/Baseline is 71.3, the claim
says 72.3") and ask for corrected claims.

**6.2 🔌 G2's support check** (1.5 h). After `check_claim`, one `call_llm` with every
remaining claim and its cited evidence. Show the cited cell *together with its row and
column names* and the whole table text, and attach figure crops. Schema:
`{verdicts: [{claim, supported, reason}]}`. A claim fails if either layer fails.

**6.3 🔌 Re-run Section 15 with the support check** (1 h). The four cases (honest / fabricated /
uncited / misattributed) on the **stub** paper, but with the real model doing the support
check. Does it now reject the misattributed claim? Save the result with `atomic_write_json`
to `results/day6_g2_boundary.json`. Either answer is a finding for the thesis.

**6.4 🔌 Real N10b** (1.5 h).
- table claims: the cited cells' boxes (unchanged code)
- text claims: `page.search_for(quote)`, which gives the sentence's exact boxes
- figure claims: compute the map for the **claim's text** (not the question's), set
  everything outside the figure region to 0, threshold it, and keep the **largest connected
  blob** (`cv2.connectedComponentsWithStats`). Its rectangle is the box.

*Why the claim's text:* the claim says *what* it is about ("the curve after epoch 10"), so
its map points more precisely than the question's map.

**6.5 🔌 Full graph on 10 dev questions** (1.5 h). Render proofs with `draw_proof(state,
pymupdf.open(pdf_path))` for 3 of them and save them to `data/debug/`. 🔋 Look at them during
the next outage: are the boxes on the right cells?

**6.6 🔌 Notebook section "Day 6: a real grounded answer"**: the report and the proof image.

### Check it works

- [ ] every claim's cites point to buffer ids (guaranteed by the enum; check anyway)
- [ ] a hand-made wrong claim (wrong value) is rejected by layer 1
- [ ] the 6.3 result is saved
- [ ] proof images: boxes on the right cells and figures for at least 3 dev questions
- [ ] `pytest -q` still green

### Write 🔋 (1 h)

Methodology updates: *Generation and verification* (the two layers of G2, the three cite
kinds) and *Attribution* (claim links instead of integrated gradients, with the reason).

### Explain it back 🔋

<details><summary>1. A claim cites a real cell with the right value, yet it is false. How?</summary>

It attaches the right number to the wrong subject, for example the high-resource row's
gain reported as the low-resource gain. The value matches, so equality tests pass; only a
check that reads which row the sentence is about can catch it.
</details>

<details><summary>2. Why are figure claims harder to verify than table claims?</summary>

A table cell has an exact text value to compare against. A figure reading ("flattens after
epoch 10") is an interpretation of pixels, and only a model that looks at the image can
judge it, which is less certain than an equality test.
</details>

### If you're behind

Use the question's map for figure boxes (skip the claim-text maps). **Never skip** 6.3.

---

## Day 7: End to end, baselines, start the big run

**Power plan.** 🔌 about 6 h, then the full-system test run, which from tonight **restarts
itself after every cut**, day and night. 🔋: the ablation reading; the Experimental Setup
text; reading the dev error analysis.

**Goal:** settings frozen on dev, the two baselines built, a power-cut-proof runner, and the
full-system test run started.

**By tonight:** `eval/frozen_config.json` · `build_graph(variant)` · `scripts/run_eval.py` ·
`scripts/resume_eval.sh` · the full-system run going.

### Understand it first 🔋

**An ablation is a controlled experiment.** To know what a part contributes, remove *only
that part* and measure again. Everything else stays identical: model, prompts, questions,
cache.

| System | Retrieval loop (G1) | Rewrite loop (G2) | Table/figure tools | Isolates |
|---|---|---|---|---|
| **Full** | ✓ | ✓ | ✓ | — |
| **Single-pass** | ✗ (one round, all tools at once) | ✗ | ✓ | the value of *agency* (loops + gates) → RQ1 |
| **Text-only** | ✓ | ✓ | ✗ | the value of *seeing* tables and figures → RQ2 |

If Full beats both, neither factor alone explains the gain.

**What exactly is single-pass?** N1 → N2 → one node that calls text, table and figure
retrieval **once each** with the original question → N8 → N9 → proofs → report. No G1, no
G2, no loops: a standard RAG pipeline built from your own parts.

**What exactly is text-only?** The full graph, but N1's sub-goals are all forced to "text",
the planner can only choose the text tool, and N9 gets no images. It *does* see the
captions and the table's text as plain text, because a text extractor would give it that.
State this in the thesis; it makes the baseline fair, not weak.

**Freezing.** After today's dev tuning, write every setting (bounds, thresholds, top-k,
prompt version, model file) into `eval/frozen_config.json` and commit it. From then on
nothing changes, and the test set is run once.

**Why a power cut cannot spoil the evaluation.** The runner writes one finished question per
line with `append_jsonl` (forced to disk), and on start it reads the file with `read_jsonl`
and skips every question already there. Every model call is in the cache. So after a cut,
the run continues at the first unfinished question, and the half-finished one is redone in
seconds from the cache. **The final results are identical to a run without cuts.**

### Steps

**7.1 🔌 Dev error analysis** (2 h). Run the full system on all 20 dev questions. For every
wrong answer, write one category:

| Category | Fix lives in |
|---|---|
| right page never retrieved | N6 / indexes |
| right page, wrong region | region choice (Day 4) |
| table parsed wrong | TATR / fallback |
| G1 too strict (abstained but evidence was there) | G1 prompt |
| G1 too lenient | G1 floor |
| N9 misread the evidence | N9 prompt |
| G2 rejected a correct claim | support-check prompt |

Fix the **top two categories only**, then stop tuning. The rest goes into the error-analysis
section of the thesis. (🔋 You can do the categorising itself during an outage from the saved
traces.)

**7.2 🔌 Variants** (2 h). In [graph.py](src/gea/graph.py): add
`cfg.VARIANT = "full" | "single_pass" | "text_only"` and `build_graph(variant)`.
`single_pass` is a new small builder; `text_only` is the full builder with the overrides
described above. Add a stub test for each variant (it runs; text-only never calls N6b/N6c).

**7.3 🔌 The runner:** `scripts/run_eval.py` (2 h).
`python scripts/run_eval.py --system full --split test --out results/runs/full_test.jsonl`

```python
done = {r["qid"] for r in read_jsonl(out)}                  # what earlier runs finished
for q in questions:
    if q["qid"] in done:
        continue                                            # resume after a power cut
    record = run_one(q, system)                             # the graph on one question
    append_jsonl(out, record)                               # forced to disk before the next one
```

- **One JSON line per question:** `qid, system, status, answer, claims, evidence` (id,
  kind, label, page, tool), `boxes, rounds, rewrites, tool_calls, model_calls, seconds,
  dag_path`. Drop numpy arrays and image bytes before writing.
- If llama-server is not reachable, the runner **waits and retries** every 30 s instead of
  recording fallback answers. A fallback answer in the test results would be a wrong
  measurement. Use `llm_ready()` before each question.
- Measure seconds per question on dev, then multiply by 150. That is how many power-window
  hours the run needs.

**7.4 🔌 The resume script:** `scripts/resume_eval.sh` (30 min). One command for after every
power cut: start llama-server in the background, wait until it answers, then run the three
systems in order. Each one skips what is already done.

```bash
#!/usr/bin/env bash
# After a power cut: start the model, wait until it answers, continue all three runs.
cd "/d/Thesis Grounded/Grounded_Explanation_Agent" || exit 1
mkdir -p data/logs
exec >> data/logs/resume_eval.log 2>&1          # unattended: everything from here goes to the log
echo "=== $(date): power is back, resuming"
source .venv/Scripts/activate
bash scripts/start_llm.sh > data/logs/llm.log 2>&1 &            # the model, in the background
until curl -sf -o /dev/null http://localhost:8080/v1/models; do sleep 5; done
for s in full single_pass text_only; do
    python scripts/run_eval.py --system "$s" --split test --out "results/runs/${s}_test.jsonl"
done
bash scripts/checkpoint.sh "evaluation progress"
```

`curl -sf` fails until the server answers "OK"; while the model is still loading it answers
503, so the loop keeps waiting. Because of `.gitattributes`, this file keeps the Unix line
endings bash needs.

**7.5 🔌 Make the PC restart the evaluation by itself** (45 min). Do this once; it handles
every cut from now on, including at night.

1. **BIOS:** restart, press `Del` (or `F2`) during start-up, and find **"Restore on AC Power
   Loss"** (on AM5 boards usually under Advanced → APM or ACPI; MSI: Settings → Advanced →
   Power Management Setup). Set it to **Power On**, then save and exit.
2. **Automatic sign-in:** download **Autologon** from Microsoft Sysinternals, run it, type your
   Windows password and press Enable. (If you sign in with a PIN, first switch off "For
   improved security, only allow Windows Hello sign-in" under Settings → Accounts →
   Sign-in options.)
3. **The task**, in Windows' **Task Scheduler** (Start menu → "Task Scheduler" → **Create
   Task…**, not "Create Basic Task"):
   - *General:* name `ResumeThesisEval`; "Run only when user is logged on".
   - *Triggers → New:* Begin the task **At log on**, your user; tick **Delay task for: 1
     minute** (so drivers and the GPU are ready).
   - *Actions → New:* Start a program. Program/script: `C:\Program Files\Git\bin\bash.exe`.
     Add arguments: `-l "/d/Thesis Grounded/Grounded_Explanation_Agent/scripts/resume_eval.sh"`.
   - *Conditions:* untick "Start the task only if the computer is on AC power".
   - Press OK. To test it without a power cut: right-click the task → **Run**, then read
     `data/logs/resume_eval.log`.

4. **Windows:** Settings → System → Power: screen and sleep **Never** while plugged in.

**7.6 🔌 Freeze** (15 min): write `eval/frozen_config.json`, then checkpoint.

**7.7 🔌 Smoke test on 3 test questions per system** (15 min): **only check that nothing
crashes**. Do not read the answers.

**7.8 🔌 Start** (as soon as 7.7 passes). Close the browser, then:

```bash
bash scripts/resume_eval.sh
```

From now on, every cut ends the same way: the power returns, the PC starts, logs in, and the
task runs this script again. Watch it live with `tail -f data/logs/resume_eval.log` (Ctrl+C
stops watching, not the run).

### Check it works

- [ ] **power-cut test:** switch the PC off at the wall in the middle of a question (or wait for the next real cut): when the power is back it starts, logs in and continues by itself, and no qid appears twice in the file
- [ ] every output line is valid JSON with all fields (`read_jsonl` loads the file)
- [ ] the runner waits, rather than writing fallback answers, while llama-server is down
- [ ] `frozen_config.json` committed *before* the test run started

### Write 🔋 (45 min)

Experimental setup: the three systems (the table above), what is frozen, dev/test
separation, why ablations rather than other systems, and one sentence on why interrupted
runs give identical results (resume + cache).

### Explain it back 🔋

<details><summary>1. Why are baselines built from your own graph better than LLaVA or plain ReAct?</summary>

With your own parts, only the one switched-off factor differs. Against a different system,
the model, prompts and retrieval all differ at once, so you could not say *why* yours did
better.
</details>

<details><summary>2. You find a prompt bug while the test run is going. What do you do?</summary>

Note it, and do **not** fix it for this run, because the test must use the frozen settings.
Report it as a limitation. If it is severe, fix it, re-freeze, re-run *all* systems on the
test set, and say in the thesis that this happened.
</details>

<details><summary>3. The power died 40 times during the evaluation. Are the results still valid?</summary>

Yes. Every finished question was forced to disk before the next started, unfinished ones were
redone from the start, and every model call returned the same cached reply. The results file
is the same as one uninterrupted run would produce. Only the timing column (`seconds`) is
affected for the interrupted questions, so report speed from uninterrupted questions.
</details>

### If you're behind

Drop to 100 test questions (state it). **Never skip** the freeze or the power-cut test.

---

## Day 8: Measure everything

**Power plan.** 🔌 the evaluation keeps running and restarting itself until all three systems
are done (check the log and the progress line below whenever you are at the PC), then the
judge (~1 h) and the metrics. 🔋 **this is the best outage day**: annotating
the 50 questions on the laptop, the metric reading, Related Work.

**Goal:** all three systems run, 50 questions annotated, every metric computed with a
confidence interval.

**By tonight:** `results/runs/*.jsonl` for all three systems · `eval/annotations.csv` ·
`src/gea/metrics.py` · `results/summary.json`.

### Understand it first 🔋

**The outcome of every question is one of three:**
**correct** · **wrong** (answered, but the answer is false: this is the *hallucination*
measure) · **abstained** (the system said it could not confirm). A good system turns
"wrong" into "abstained" or "correct". Report all three, never accuracy alone.

**Accuracy judge.** For each answered question, the model compares the system's answer with
SPIQA's gold answer and returns `{correct: bool, reason}`. Because a model judges, you
**check the judge**: grade 30 answers yourself and report how often you agree. Cohen's
kappa is agreement corrected for luck: 1 = perfect, 0 = no better than chance.

**The four metrics of your thesis, with toy examples:**

| Metric | Question it answers | Toy example | RQ |
|---|---|---|---|
| **RF** (Retrieval Faithfulness) | Of the visual evidence it fetched, how much was the right figure/table? | buffer has Table 3, Figure 1, Figure 2; gold is Table 3 → RF = 1/3 | RQ1, RQ2 |
| **MAP** (Modality Attribution Precision) | Do claims cite the right source? | 4 claims, 3 cite the right item → 0.75 (hand-checked on the 50) | RQ2 |
| **AES** (Agent Efficiency Score) | Needed hops ÷ tool calls actually made | needed 2, used 4 → 0.5 (capped at 1) | RQ1 |
| **CGS** (Cross-modal Grounding Score) | How well does the box overlap the true region? (IoU) | two 10×10 boxes overlapping by half → 50 / 150 = 0.33 | RQ3 |

Also report the **hit rate**: was the gold figure/table in the buffer at all?

**IoU** (Intersection over Union) = overlap area ÷ combined area. 1 = identical boxes,
0 = no overlap. "IoU ≥ 0.5" is the usual "good enough" line.

**Bootstrap confidence interval: is the difference real or luck?** With 150 questions, a
2-point difference might be chance. The bootstrap answers this: draw 150 questions *with
replacement* from your 150, compute "Full minus Baseline" on that draw, and repeat 10,000
times. If 95 % of those differences are above 0, the improvement is unlikely to be luck.

### Steps

**8.1 🔌 Keep the evaluation going.** The Day 7 task restarts it after every cut. If you did not
set that up, run this yourself each time the power returns:

```bash
bash scripts/resume_eval.sh
```

Check progress anytime: `python -c "from gea.safeio import read_jsonl; print(len(read_jsonl('results/runs/full_test.jsonl')))"`.

**8.2 🔋 Annotate 50 questions** (2–3 h: *on a laptop during outages, or on the desktop while
the evaluation runs*; Paint does not need the GPU). Prepare in a power
window: choose 50 test questions with `random.Random(42)`, 25 table and 25 figure
references, **before** looking at any system output. Render each reference's page at
**144 dpi** (pixel = point × 2) into `data/annotate/`, and copy that folder to the laptop if
you have one. Then:
1. Open the PNG in **Paint**: it shows the mouse position in pixels at the bottom.
2. For a table, box the cell(s) that answer the question; for a figure, box the part of the
   chart that answers it.
3. Write `qid, page, x0, y0, x1, y1` (pixels ÷ 2 = points) into `annotations.csv`. Save
   after every row (Ctrl+S), and checkpoint after every 10 rows: the power, or a laptop's
   battery, can go at any second.

Copy `annotations.csv` back to `eval/annotations.csv` and commit it. Once all runs are done,
also mark for these 50 whether each published claim cites the right source (for MAP), which
is more outage work.

**8.3 🔌 The judge** (1 h + run time). `scripts/judge.py` writes `correct: true/false` for every
answer into `results/judged/<system>_test.jsonl`. It is resumable with the same
`read_jsonl` / `append_jsonl` pattern, and every judge call is cached. Then 🔋 grade 30
random answers yourself, without looking at the judge's verdict (print them on paper or
copy to the laptop), and compute agreement and kappa.

**8.4 🔌 `src/gea/metrics.py`** (2 h):

```python
def rf(visual_labels, gold):                 # None when nothing visual was retrieved (text-only)
    return sum(l == gold for l in visual_labels) / len(visual_labels) if visual_labels else None

def aes(needed_hops, tool_calls):
    return min(1.0, needed_hops / tool_calls) if tool_calls else 0.0

def bootstrap_diff(a, b, n=10_000, seed=42):   # a, b: per-question 0/1, same questions, same order
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(diffs.mean()), np.percentile(diffs, [2.5, 97.5])
```

`needed_hops` = 1 for a normal SPIQA question, or 2 when it is in the multi-hop subset.
`tool_calls` counts N6a–N6c only (N7 is automatic, not a choice). CGS uses `iou()` from
[proofs.py](src/gea/proofs.py). An abstention scores CGS = 0; also report CGS over answered
questions only.

**8.5 🔋 Multi-hop subset** (30 min). Read only the *questions* (not the results) and tag the
ones that need narrative text **and** a figure or table. Report every metric for that
subset too: it is where the loop should matter most.

**8.6 🔌 `results/summary.json`** (1 h, written with `atomic_write_json`): every metric × every
system × {all, multi-hop}, with bootstrap intervals for Full vs each baseline.

**8.7 🔋 Related Work** (any outage time). A short chapter with four parts: multimodal document
QA (SPIQA, DocVQA, ChartQA) · visual retrieval (ColPali, ColBERT, VisRAG, M3DocRAG) · agentic
and corrective RAG (ReAct, Self-RAG, Corrective RAG) · attribution and faithfulness (ALCE,
RAGAS, integrated gradients). **Download those papers' PDFs to the laptop beforehand** so you
can read them offline. Check every reference's details on the paper's page before adding it
to `references.bib`.

### Check it works

- [ ] the metric functions pass tests that use the toy examples above (write them!)
- [ ] three run files, 150 lines each (or the number you froze), no duplicate qids
- [ ] judge agreement measured on 30 answers

### Explain it back 🔋

<details><summary>1. System A: 60 % correct, 30 % wrong, 10 % abstained. System B: 58 % correct, 12 % wrong, 30 % abstained. Which is better for researchers?</summary>

Probably B. Its accuracy is about the same, but it is wrong far less often; it says "I
could not confirm" instead. A researcher can act on a stated gap but cannot detect a
confident error. This is the core argument of the thesis, and the 3-way breakdown is how
you show it.
</details>

<details><summary>2. Why choose the 50 annotation questions before looking at the outputs?</summary>

If you choose after looking, you might (even without meaning to) pick questions where the
boxes look good, which biases CGS upwards.
</details>

### If you're behind

Annotate 30 instead of 50 (state it). If the evaluation is not finished tonight, it continues
into Day 9's power windows; Day 9's writing does not need it until the afternoon. **Never
skip** the judge-agreement check.

---

## Day 9: Results into the thesis

**Power plan.** 🔌 about 3 h: finishing any evaluation that spilled over, generating tables and
figures, building the PDF. 🔋 **most of the day**: Results, Discussion and Implementation
chapters on the laptop.

**Goal:** the thesis contains the real results, matches the implementation everywhere, and
no number in it was typed by hand.

**By tonight:** `scripts/make_thesis_tables.py` · Results and Discussion chapters · all old
chapters consistent · the PDF builds.

### Understand it first 🔋

**Generated, not typed.** Like notebook test 11I: a script reads
`results/summary.json` and writes `paper/tables/*.tex` and `paper/figures/*.pdf`. The thesis
uses `\input{...}` and `\includegraphics{...}`. If a number changes, you re-run one script.
(During an outage you can write the text around a table before the table exists: leave the
`\input{tables/tab_main_results}` line in, and it fills in at the next build.)

**Answer the research questions directly.** Each RQ gets its own subsection that opens with a
one-sentence answer, followed by the evidence:

| RQ | Main evidence |
|---|---|
| RQ1: does the loop improve accuracy and reduce hallucination? | correct / wrong / abstained for Full vs Single-pass (all + multi-hop), AES, bootstrap |
| RQ2: can visual embeddings fetch the right tables and figures? | page hit@k (Day 3), RF, hit rate, Full vs Text-only, MAP |
| RQ3: can the agent produce precise boxes? | CGS (mean IoU, % ≥ 0.5), proof images, the G2-boundary result (6.3) |

**Honest reporting.** A result that did not go your way still goes in, explained. Threats to
validity to list: one annotator · the judge is the same model family as the system · only CS
papers · a 4-bit model · within-paper retrieval · 150 questions · RQ3's "human
verifiability" was not measured with a user study.

### Steps

**9.1 🔌 `scripts/make_thesis_tables.py`** (2 h):
- `tab_main_results.tex`: systems × {correct, wrong, abstained, RF, AES, CGS, calls per
  question, seconds per question}
- `fig_outcomes.pdf`: stacked bars of correct / wrong / abstained per system (the key figure)
- `fig_cgs_hist.pdf`: the IoU distribution
- `tab_multihop.tex`: the same metrics on the multi-hop subset

**9.2 🔋 Fix the existing chapters** (2 h; pull the repo onto the laptop first). The exact places:

| Where | Change |
|---|---|
| [main.tex](paper/main.tex) | `\usepackage{graphicx}`; abstract in past tense with real numbers; `\input` the new chapters |
| [methodology.tex:95](paper/methodology.tex#L95) | three tools; RegionZoom is triggered by figure size (decision 7) |
| [methodology.tex:226](paper/methodology.tex#L226) | Model Choice: Qwen2.5-VL-7B, 4-bit, llama.cpp; the reason is local / reproducible / fits 12 GB, no longer gradients |
| [methodology.tex:245](paper/methodology.tex#L245) | Text token attribution → claim links from verified citations (decision 2) |
| [methodology.tex:287](paper/methodology.tex#L287) | Baselines → the two ablations (decision 3) |
| methodology.tex, Dataset section | 150 test / 20 dev / 50 annotated; coverage and dropped questions |
| [architecture.tex:57](paper/architecture.tex#L57), line 70 | RegionZoom; Qwen2.5-VL; remove "gradients" |
| [Graph.tex:67](paper/Graph.tex#L67) | node label "N10a Claim Links / verified citations" |
| [Graph.tex:205](paper/Graph.tex#L205) | the running example uses N6b (table), not N6c + zoom |
| [Graph.tex:250](paper/Graph.tex#L250) | N10a paragraph |
| Graph.tex, caption of Figure 4.1 | the N1 → N5 arrow is data flow; execution order is N1 → N2 → N5 |
| [references.bib](paper/references.bib) | add Qwen2.5-VL (Bai et al., 2025, arXiv:2502.13923), llama.cpp, PyMuPDF and the Related Work papers |

**9.3 🔋 New chapters** (4 h). Every number so far is in
[notes/results_for_paper.md](notes/results_for_paper.md), grouped by chapter with its source.
*Implementation* (from `notes/journal.md`, including the stub-first
method, the power-cut-safe design, the execution profile table `tab_execution_profile.tex` and
the trace figure `fig_execution_trace.pdf`), *Results* (per RQ), *Discussion* (error analysis
from 7.1, the G2 boundary, limitations).

**9.4 🔌 Build and read** (1 h). No `??` (undefined references), no overfull warnings in
tables, every figure readable in black and white. Push.

### Check it works

- [ ] every number in Results comes from an `\input` file
- [ ] searching the `.tex` files for "Qwen2-VL", "integrated gradients" and "LLaVA" finds only intended mentions
- [ ] the PDF builds cleanly

### If you're behind

Write Discussion as bullet points first, then turn them into prose tomorrow.

---

## Day 10: Finish line

**Power plan.** 🔌 about 3 h: the reproducibility check and the final build. 🔋: the conclusion,
the read-through, the slide outline and defence practice.

**Goal:** a finished, reproducible thesis, and you can defend every part of it.

### Steps

**10.1 🔋 Conclusion and abstract** (2 h). The conclusion answers each RQ in one paragraph,
then gives the contributions and future work (integrated gradients with a larger GPU, a user
study for verifiability, more domains than CS).

**10.2 🔌 Reproducibility** (2 h):
- `README.md`: what the project is, setup (point to Day 1), and the commands that regenerate
  the indexes, the runs, the scores and the thesis tables, in order.
- `uv pip freeze > requirements-lock.txt` (the exact versions that produced the results).
- **Delete-and-rebuild check:** rename `data/llm_cache` and re-run 5 test questions. Are the
  answers the same? Write down what you find (GPU maths can differ slightly; that is a
  normal limitation to mention).
- `git tag v1.0-thesis` and push.
- **Undo the Day 7 automation:** in Task Scheduler, right-click `ResumeThesisEval` →
  Delete; run Autologon again and press Disable. You can keep the BIOS setting.

**10.3 🔋 Final read-through** (2 h; a printout works without power): read the PDF from start
to end, out loud for the abstract and conclusion. Check that every figure is referenced in
the text.

**10.4 🔋 Defence preparation** (2 h): a 12-slide outline (problem, idea, architecture, the two
gates, one live-looking example with the proof image, the three systems, the outcome chart,
CGS, the G2 boundary, limitations, contributions, thanks), then practise Appendix A out
loud.

### Check it works

- [ ] a fresh clone + Day 1 setup + `pytest -q` passes
- [ ] `make_thesis_tables.py` regenerates exactly the committed tables
- [ ] `v1.0-thesis` tag pushed

---

## Appendix A: Defence practice

Answer each one out loud in under a minute, then compare.

<details><summary>Why not just put the whole paper into the model's context?</summary>

Three reasons. A full paper with page images exceeds what a 7B model handles well on 12 GB.
Long contexts make models miss details ("lost in the middle"). And it still would not say
*where* each claim came from. Retrieval plus citations gives a small, checkable evidence set.
</details>

<details><summary>Why search page images instead of extracted text?</summary>

Text extraction breaks exactly where scientific papers are dense: multi-column headers, merged
cells, axis labels. ColQwen2 sees the layout as it is printed. The text index is still used for
narrative claims, so you get both.
</details>

<details><summary>Why two gates instead of one?</summary>

They catch different mistakes at different times. G1 (before writing) prevents answering with
missing evidence. G2 (after writing) catches using the evidence wrongly. A single check at the
end could not tell "the table was never found" from "the table was misread".
</details>

<details><summary>Doesn't the system just abstain a lot to look safe?</summary>

That is why the results report correct / wrong / abstained separately, and AES for efficiency.
A system that abstained on everything would have 0 wrong answers *and* 0 correct ones; the
table shows that immediately.
</details>

<details><summary>The judge is a language model. How do you know it is right?</summary>

I graded 30 answers myself and report agreement and Cohen's kappa. Its bias (same model family
as the system) is listed as a threat to validity.
</details>

<details><summary>Why did you drop integrated gradients?</summary>

It needs gradients through the generator. The full-precision 7B model is about 14 GB, more than
the B580's 12 GB, and the 4-bit llama.cpp model cannot compute gradients at all. More
importantly, G2 already links every claim to its exact source, which is exact rather than
estimated. Integrated gradients is future work.
</details>

<details><summary>What can G2 not catch?</summary>

With equality tests only: a real number attached to the wrong subject (Section 15). The model
support check was added for that; my result in 6.3 shows how often it catches it. Figure
readings remain model-judged, which is less certain than an equality test.
</details>

<details><summary>Is 150 questions enough?</summary>

The bootstrap confidence intervals answer that directly: they show which differences are
larger than chance at this size. Where an interval crosses zero, I do not claim a difference.
</details>

<details><summary>Does 4-bit quantisation hurt?</summary>

Slightly, in general. It is the price of running locally on 12 GB. All three systems use the
same quantised model, so the *comparison* between them is fair.
</details>

<details><summary>Your runs were interrupted by power cuts. Can we trust the numbers?</summary>

Yes. Each finished question was written to disk before the next one started, an interrupted
question was redone from the start, and every model reply came from a cache keyed on the exact
request, so a resumed run produces the same results as an uninterrupted one. The tests in
`tests/test_safeio.py` check the power-cut cases directly.
</details>

<details><summary>Why are the baselines your own graph and not published systems?</summary>

To change one factor at a time. With the same model, prompts and data, any difference comes
from the loop (single-pass) or the visual tools (text-only), not from a different model.
</details>

<details><summary>How were the boxes evaluated?</summary>

50 questions chosen at random before seeing any output, boxed by hand on the reference page;
CGS is the IoU between the system's box and mine. One annotator is a limitation.
</details>

<details><summary>What would you do with more time?</summary>

A user study on verifiability (do readers check answers faster and more accurately with
boxes?), integrated gradients on a larger GPU, other domains such as biomedicine, and
retrieval across many papers instead of within one.
</details>

---

## Appendix B: Glossary

| Term | Plain meaning |
|---|---|
| **Abstain** | the system says "I could not confirm this" instead of answering |
| **Ablation** | removing one part to measure what it contributed |
| **Adapter** | a small add-on layer that specialises a model (SPECTER2's query vs document adapters) |
| **AES** | needed retrieval hops ÷ tool calls made; efficiency |
| **Agent** | a program that loops: think → act → look |
| **Atomic write** | write to a temporary file, then rename; a power cut leaves either the old or the new file, never half |
| **bbox** | a box `(x_min, y_min, x_max, y_max)` on a page, in points |
| **bf16** | a 16-bit number format; half the memory of 32-bit, nearly the same accuracy for models |
| **Bootstrap** | resampling your questions many times to see how stable a difference is |
| **CGS** | mean IoU between predicted and hand-drawn boxes |
| **Checkpoint** | `bash scripts/checkpoint.sh "…"`: commit + push to GitHub; your work is only safe once pushed |
| **Chunk** | a ~150-word piece of paper text, the unit of text search |
| **Citation (cite)** | the evidence id plus the quote, cell or observation a claim rests on |
| **ColQwen2 / ColPali** | a model that embeds page *images* as one vector per patch |
| **Conditional edge** | a LangGraph signpost: reads the state and returns the next node's name |
| **Cosine similarity** | how much two vectors point the same way (1 = same) |
| **Dev set** | questions used for tuning; never the test questions |
| **dpi** | dots per inch; pixels = points × dpi / 72 |
| **Embedding** | a list of numbers that represents meaning; similar meaning = close vectors |
| **Evidence buffer** | N8's folder of everything retrieved; the only thing N9 may read |
| **FAISS** | a library for fast nearest-vector search |
| **Git Bash** | the Linux-style terminal that comes with Git for Windows; `D:\` is `/d/` |
| **Fan-out** | one node starting several nodes at once (N10a–c) |
| **fsync** | forcing written data out of memory onto the disk, so a power cut cannot lose it |
| **G1 / G2** | Sufficiency Gate (before writing) / Faithfulness Gate (after writing) |
| **GGUF** | llama.cpp's model file format, usually quantised |
| **Hallucination** | a confident statement not supported by the source |
| **IoU** | overlap area ÷ combined area of two boxes |
| **JSON schema** | a description of the allowed answer shape; llama.cpp enforces it |
| **JSONL** | one JSON record per line; easy to append to and to resume |
| **Kappa (Cohen's)** | agreement between two graders, corrected for chance |
| **LangGraph** | the library that runs your nodes and edges |
| **llama.cpp / llama-server** | the program that runs the GGUF model on the B580 and serves an API |
| **LLM-as-judge** | using a language model to grade answers or evidence |
| **MAP** | share of claims that cite the correct source |
| **MaxSim / late interaction** | for each query word, its best patch score, summed over the words |
| **mmproj** | the vision part of a multimodal GGUF model |
| **Patch** | a small square of a page image; ColQwen2 gives each one a vector |
| **PDF point** | 1/72 inch; PDF coordinates use it |
| **Quantisation** | storing model weights with fewer bits to save memory |
| **RAG** | retrieve evidence, then generate the answer from it |
| **ReAct** | Reason + Act: the think-act-observe loop |
| **Reducer** | the LangGraph rule for merging two writes to one key (e.g. append) |
| **Resumable** | a job that skips finished items when restarted |
| **RF** | share of retrieved visual evidence that is the gold figure/table |
| **Similarity map** | per-patch match scores laid out as a heat map over the page |
| **SPECTER2** | a scientific text embedding model |
| **SPIQA** | the dataset: questions about CS papers tied to a figure or table |
| **State** | LangGraph's shared whiteboard (`AgentState`) |
| **Stub** | a stand-in function with the right shape but fake logic |
| **Superstep** | one LangGraph step; nodes in the same step run in parallel |
| **Table Transformer (TATR)** | a detector that finds a table's rows and columns |
| **Temperature** | randomness of the model's choices; 0 = always the most likely |
| **xpu** | PyTorch's name for Intel GPUs like the B580 |

---

## Appendix C: Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.__version__` ends in `+cpu` | requirements installed torch before the xpu build | `uv pip install --reinstall torch torchvision --index-url https://download.pytorch.org/whl/xpu` |
| `torch.xpu.is_available()` is False | old driver, or VS Code started before the driver update | update the Intel driver, reboot |
| `UnicodeEncodeError: 'charmap' codec` | Windows console encoding | `setx PYTHONUTF8 1`, restart VS Code |
| C: fills up | a cache still on C: | check `echo $HF_HOME $UV_CACHE_DIR`; if empty, Day 1 step 1.2 and restart VS Code |
| uv warns "Failed to hardlink files; falling back to full copy" | uv's cache is on a different drive (C:) from `.venv` (D:): the terminal was opened before `setx UV_CACHE_DIR` | harmless for `.venv` (it got full copies); free C: with `uv cache clean --cache-dir "C:/Users/tam1m/AppData/Local/uv/cache"`, and use a new terminal from now on |
| `$'\r': command not found` when running a `.sh` script | the script got Windows line endings | `sed -i 's/\r$//' scripts/*.sh`; `.gitattributes` prevents it for files from git |
| `python` shows no `>>>` prompt and seems stuck | the separate Git Bash window cannot show Python's prompt | use `winpty python`, or VS Code's terminal |
| `source: .venv/Scripts/activate: No such file` | not in the project folder | `cd "/d/Thesis Grounded/Grounded_Explanation_Agent"` first |
| a Windows program reads `/something` as a file path | Git Bash converts arguments that look like paths | put `MSYS_NO_PATHCONV=1` in front of that command |
| `download_all.py` says STOP, cache on C: | `HF_HOME` not seen by this terminal | open a new terminal (it was set while the old one was open) |
| some PDFs failed to download | internet dropped, or arXiv refused | run `download_all.py` again later; finished files are skipped |
| downloads warn "hf_xet is not installed" or "symlinks … not supported" | Hugging Face's faster download tool is missing; Windows blocks symlinks without Developer Mode | both harmless: the normal download is used, and the cache stores plain files (each model still only once) |
| a model "cannot be found" or hangs while loading | internet down and `HF_HUB_OFFLINE` not set | `setx HF_HUB_OFFLINE 1` (Day 1, 1.9); for a new download, `HF_HUB_OFFLINE=0 python scripts/download_all.py` |
| notebook can't find `gea` | wrong kernel, or `-e .` not installed | choose the `.venv` kernel; `uv pip install -e .` |
| "requires the ipykernel package" | VS Code picked another Python | choose the `.venv` kernel (top right) |
| notebook variables gone after a power cut | the kernel died with the PC | Run All again; heavy steps load their saved files |
| `start_llm.sh`: "Missing …gguf" | the GGUF download is not finished | run `download_all.py` again |
| `llama-server: command not found` or `start_llm.sh`: "llama-server not found" | llama.cpp is not unzipped in `D:\tools\llama.cpp` | Day 1, step 1.7; call it by its full path `/d/tools/llama.cpp/llama-server.exe` |
| `start_llm.sh`: "The GPU already has … MB in use" | another program (often another project's training) is using the B580 | let it finish or stop it; meanwhile do the no-GPU steps (see "The B580 budget") |
| llama-server: out of memory (`ErrorOutOfDeviceMemory`) | another program on the GPU, context too big, or ColQwen2 also on the GPU | stop other GPU programs; `bash scripts/start_llm.sh 6144`; ColQwen2 on the CPU |
| llama-server slow | Vulkan build on Arc | try the SYCL release build; keep `-np 1` |
| model replies are not valid JSON | `response_format` missing | always go through `call_llm` |
| `read_jsonl` raises "line N is damaged" | a damaged line in the middle (not a power cut, which only hits the last line) | open the file, delete that one line, re-run: the runner redoes that question |
| files named `*.tmp.*` in `data/` | a power cut during a write | harmless; `remove_leftovers()` (run by `download_all.py`) deletes them |
| git: "object file is empty" or "corrupt" after a cut | a git write was interrupted | `git fsck` shows the damage; the safe fix is to clone the repo again from GitHub into a new folder and copy your changed files over (that is why you checkpoint after every step) |
| `checkpoint.sh`: "push failed" | no internet | nothing is lost: it is committed on the PC; run it again when the internet is back |
| `ModuleNotFoundError` or "DLL load failed" after a cut during installing | half-installed packages in `.venv` | delete the `.venv` folder and repeat Day 1, step 1.3; uv's cache makes it quick |
| VS Code: "Unreadable Notebook" after a cut | the notebook was being saved at that moment | right-click the file → Open Timeline and pick the last good version, or `git checkout -- notebooks/langgraph_demo.ipynb` |
| the PC stays off when the power returns | the BIOS setting was not saved | Day 7, step 7.5 (1) again; make sure you chose "Save & Exit" |
| the PC starts but the evaluation does not continue | autologon or the task is not working | Task Scheduler → ResumeThesisEval → History; read `data/logs/resume_eval.log` |
| Windows runs "scanning and repairing drive" after a cut | the disk was written to when the power went | let it finish; then `git status` and `pytest -q` |
| colpali-engine import / attribute error | version differences | keep the pins in `requirements.txt`; check names with `help()` |
| figure boxes land in the wrong place | rows/columns swapped in the similarity map | Day 4 step 4.2 |
| `find_tables()` finds nothing | tables without ruling lines | try `find_tables(strategy="text")`, or rely on TATR |
| the evaluation stopped and did not restart | power cut without the Day 7 automation, or llama-server down | `bash scripts/resume_eval.sh`; it resumes |
| MiKTeX `latexmk` asks for Perl | MiKTeX's latexmk needs Perl | use the pdflatex/bibtex sequence (Day 1, 1.8) |
