# Cost Optimization and Hybrid Inference

Gem provides tools to help you track, limit, and optimize your cloud model costs.

## Cost Tracking

You can view your current usage statistics and model cost breakdown using:
```bash
Gem cloud cost
```

### Output Example
```
--- Cloud Cost Statistics ---
Daily Usage:   $0.0250 / $5.0000
Monthly Usage: $0.1500 / $50.0000

Model Breakdown:
  - google/gemma-2-9b-it:free: $0.0000
  - meta-llama/llama-3-70b-instruct: $0.1500
```

## Hybrid Inference Strategy

To optimize costs, you can use a hybrid approach where simple queries use local inference (e.g., local Gemma or Qwen models), and complex queries are routed to larger cloud models.

### Example Workflow
1. Start with a local model: `Gem chat --model mlx-community/gemma-4-e2b-it-4bit`
2. If the request is complex, explicitly use a cloud model or configure your application's agent-to-agent logic to offload specific tasks to OpenRouter.

### Cost-Aware Routing
In your `CostTracker` module, when you approach your daily limit (e.g., 80%), Gem logs a warning. You can programmatically intercept these warnings to downgrade the selected model or switch to local inference.
