# Cloud decisions with Gemini Decision 1

Google's `gemini-decision-1` answers System One questions (`noul`, `choice`, `score`) in one pass over text, images, audio, video and PDFs. Its wire format matches basalt's own `/v1/systemone`, so it works as a cloud stage in front of a local decision model:

```bash
export GEMINI_API_KEY=...        # read from the environment, never a flag or a file
basalt serve --decision-model cloud:gemini/gemini-decision-1 --decision-escalate self
```

- Unsure questions (confidence below `--decision-escalate-below`), failed calls and rate limits go to the local model.
- `/v1/systemone` takes `"media": [{"kind": "image|audio|video|pdf", "mime_type": "...", "data": "<base64>"}]` next to `state`. Media is refused when no stage can read it.
- In a compose file, the model is `source: cloud:gemini/gemini-decision-1` with `api_key_env: GEMINI_API_KEY`.
  - Gemini models serve decide cascades only.
  - Pipelines keep perceived text on this Mac unless that model sets `data: media`. Then the attached files themselves, and their perceived text, go to Google for the decision.
- Cost is counted from the token usage that the API reports, at $0.10 per million input tokens, and counts toward compose budgets.

## A calibration teacher

 Gemini Decision 1 returns calibrated probabilities, not just labels. Google reports that it leads on probability calibration. That makes it an ideal calibration teacher for basalt's local decision models: run the same states through it and through a local readout, and fit the local temperature, or train an OpenJev head, against its probabilities. Then the local decider's confidence means what the cloud's does, and escalation thresholds carry over. Wiring this into the calibration step is planned with #423's local JEV work (J6: calibration read from the bundle).
