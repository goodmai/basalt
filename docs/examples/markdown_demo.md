# Markdown Rendering Demo

This demonstrates the **Epic 16.3** markdown rendering capabilities.

## Features

### Syntax Highlighting

Here's some Swift code:

```swift
func factorial(_ n: Int) -> Int {
    if n <= 1 { return 1 }
    return n * factorial(n - 1)
}

print(factorial(5)) // 120
```

### Python Example

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print([fibonacci(i) for i in range(10)])
```

### Lists

Unordered list:
- Terminal colors (green, red, yellow, blue)
- Table rendering (Unicode, ASCII, Minimal)
- **Markdown support** with *syntax highlighting*
- Code blocks with language detection

Ordered list:
1. Parse markdown with swift-markdown
2. Syntax highlight with Splash
3. Render to terminal with ANSI colors
4. Support multiple languages

### Blockquotes

> "The best way to predict the future is to invent it."
> — Alan Kay

### Links

Visit [GitHub](https://github.com) for more information.

### Inline Formatting

This is **bold**, this is *italic*, and this is `inline code`.

### JSON Example

```json
{
    "name": "GemmaServer",
    "version": "0.6.0",
    "features": [
        "Terminal UI",
        "Table Rendering",
        "Markdown Support"
    ]
}
```

### JavaScript

```javascript
const greet = (name) => {
    console.log(`Hello, ${name}!`);
};

greet("World");
```

## Summary

✅ Full markdown support  
✅ Syntax highlighting for Swift, Python, JavaScript, JSON, Bash  
✅ Beautiful terminal rendering  
✅ Color-coded output  
