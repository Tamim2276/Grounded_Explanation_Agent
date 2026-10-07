"""Grounded Explanation Agent -- the code behind notebooks/langgraph_demo.ipynb.

The notebook explains and tests; this package holds the code. BUILD_PLAN.md says
which file each day of the plan changes.

    config      every constant and switch           question    N1, N2
    device      the Intel Arc B580 (torch "xpu")    planner     N5
    state       AgentState, Evidence                retrieval   N6a-N7, N8
    trace       the one-line trace + events log     sufficiency G1 and the retrieval loop
    text        terms, sentences, score             generation  N9, G2 and the regenerate loop
    stub_paper  the fake paper + example questions  proofs      N10a-N10c, N11, N12
    corpus      N3, N4 and get_corpus()             graph       the graph builders
    llm         the local model (llama.cpp)         profiling   cost table, scenarios
    viz         show(), graph and proof drawings
"""

__version__ = "0.1.0"
