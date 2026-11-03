#!/bin/zsh
base=${1:-origin/main}
git diff --unified=0 "$base" -- '*.swift' | awk '
  /^\+\+\+ b\// { file = substr($0, 7); next }
  /^@@/ { split($3, a, ","); line = substr(a[1], 2) + 0; next }
  /^\+/ { printf "%s:%d: %s\n", file, line, substr($0, 2); line++ }
' | grep -E '@unchecked Sendable|nonisolated\(unsafe\)|NSLock|NSRecursiveLock|NSCondition|Mutex<|withLock|Thread \{|Thread\(|DispatchQueue|DispatchSemaphore|\.wait\(\)|static var|CompiledTrace|MLX\.compile|compile\(|asyncEval|Stream\.defaultStream|synchronize\(\)|container\.perform|\.next\(\)|TokenIteratorProtocol|KVCache|MambaCache|ArraysCache|\.copy\(\)'
