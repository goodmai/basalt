import re
import os

with open("docs/features.md", "r", encoding="utf-8") as f:
    content = f.read()

clusters = re.split(r'### (Кластер \d+: .+)', content)
# clusters[0] is intro, clusters[1] is "Кластер 1...", clusters[2] is table content for C1, etc.

mapping = {
    1: ("✅ Done", "OnboardingFlow.swift, OnboardCommand.swift, ServerConfig.swift"),
    2: ("✅ Done", "TokenBudgetCalculator.swift, ContextDegradationProfiler.swift, PromptContextBuilder.swift"),
    3: ("⚠️ Partial", "MLXInferenceEngine.swift, TerminalManager.swift"),
    6: ("✅ Done", "TerminalManager.swift"),
    8: ("✅ Done", "MCPServer.swift"),
    9: ("✅ Done", "ModelOrchestratorActor.swift, AgentCapabilityAnalyzer.swift"),
    10: ("✅ Done", "ProgressBar.swift, Spinner.swift, TerminalStatus.swift"),
    11: ("✅ Done", "AuthService.swift, JWTAuthenticator.swift, PrivacyInfo.xcprivacy"),
    14: ("✅ Done", "CostTracker.swift"),
    16: ("✅ Done", "AgentsCommand.swift"),
    18: ("✅ Done", "TableRenderer.swift, MarkdownRenderer.swift, DiffRenderer.swift, TerminalUI.swift"),
    19: ("✅ Done", "AuthController.swift, RESTServer.swift"),
    20: ("✅ Done", "SystemProfiler.swift, ModelFitAnalyzer.swift"),
    23: ("✅ Done", "ServeCommand.swift, RESTServer.swift"),
    26: ("✅ Done", "ModelRouter.swift, ModelsCommand.swift"),
    28: ("✅ Done", "GemError.swift"),
    29: ("✅ Done", "TerminalUI.swift"),
    30: ("✅ Done", "CloudAPIClient.swift, OpenRouterClient.swift")
}

os.makedirs("docs/specs", exist_ok=True)

for i in range(1, 31):
    idx = (i * 2) - 1
    if idx >= len(clusters):
        break
    
    header = clusters[idx]
    body = clusters[idx+1]
    
    status, files = mapping.get(i, ("❌ Planned", "—"))
    
    # Process table to add status and files
    rows = body.strip().split("\n")
    processed_rows = []
    for row in rows:
        if "|" in row and not "Функция (Feature)" in row and not "| :---" in row:
            # Extract feature name and description
            parts = [p.strip() for p in row.split("|")]
            if len(parts) >= 5:
                # parts[0] is empty, parts[1] is No, parts[2] is Name, parts[3] is Desc, parts[4] is UseCase
                processed_row = f"| {parts[1]} | {parts[2]} | {status} | {files} | {parts[3]} |"
                processed_rows.append(processed_row)
    
    with open(f"docs/specs/checklist{i}.md", "w", encoding="utf-8") as cf:
        cf.write(f"# {header}\n\n")
        cf.write("| № | Функция (Feature) | Статус | Детали реализации | Что сделано |\n")
        cf.write("| :--- | :--- | :---: | :--- | :--- |\n")
        for row in processed_rows:
            cf.write(row + "\n")

print("Created 30 checklists.")
