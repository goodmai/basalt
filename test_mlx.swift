import MLX
import Foundation

print("Checking MLX Metal support...")
let a = MLXArray(0..<10)
let b = MLXArray(repeating: 1, count: 10)
let c = a + b
print("Result: \(c)")
print("Device: \(Device.defaultDevice())")
