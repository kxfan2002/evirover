"""Multi-turn agentic evaluation.

The QA path (run_eval default) strips tools and scores a single <answer>. This
package adds the agentic path the model was actually SFT'd on: the full tool
system prompt, a <tool_call> -> execute -> <tool_response> loop with image
feedback, ending in an <answer> that the *same* scorers (eval.scorers) grade.

Public entry point: `run_agent_sample(sample, client, tools, cfg) -> row` which
returns a per-sample dict in the exact schema report.py expects, plus a saved
trajectory.
"""
