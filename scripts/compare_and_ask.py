import json
from agent.llm_runner import run_agent

with open("mlflow_metrics.json", "r") as f:
    mlflow_metrics = json.dumps(json.load(f), indent=2)

with open("arize_metrics.json", "r") as f:
    arize_metrics = json.dumps(json.load(f), indent=2)

response = run_agent(
    mlflow_metrics=mlflow_metrics,
    arize_metrics=arize_metrics,
    prompt_template="agent/prompt_template.txt"
)

print("🧠 Agent Response:\n")
print(response)
