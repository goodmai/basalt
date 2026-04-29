# OpenRouter Setup

To connect Gem with OpenRouter for cloud model inference, you must configure your API key.

## 1. Get an API Key
Go to [OpenRouter Keys](https://openrouter.ai/keys) and generate a new key.

## 2. Configure Gem
Run the following command to configure your key and budget limits:

```bash
Gem cloud configure
```

You can also pass arguments directly:
```bash
Gem cloud configure --api-key sk-or-v1-... --daily-budget 5.0 --monthly-budget 50.0
```

Alternatively, you can provide the key via environment variable:
```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
Gem cloud configure
```

## Security Best Practices
- **API Key Storage:** Gem stores the API key in `~/.gem/cloud.json` with `0600` permissions (only accessible to the user).
- **Environment Variables:** For CI/CD environments, prefer passing `OPENROUTER_API_KEY` directly via environment variables without creating the configuration file.
- **Budgeting:** Always set a daily and monthly budget to prevent unexpected costs.
