import os

replacements = {
    "GemmaServerCore": "GemCore",
    "GemmaServerTests": "GemTests",
    "GemmaServerBin": "GemBin",
    "GemmaServerError": "GemError",
    "GemmaServerCLI": "GemCLI",
    "GemmaServer": "gem",
    "com.gemmaserver": "com.gem",
    "gemmaserver": "gem",
    "gemma": "gem", # Might be risky, let's be careful, gemma-4 model name could be changed to gem-4. 
}

# Wait, replacing "gemma" with "gem" will break model names like "gemma-4-31b". 
# Let's avoid "gemma" -> "gem". The instruction only says "программа называется gem".
# So GemmaServer -> gem.
