import MLX
import Foundation

@main
struct mlx {
    static func main() {
        // Create a simple array
        let a = MLXArray(0..<4, [2, 2])
        let b = MLXArray.ones([2, 2])
        
        // Perform an operation
        let c = a + b
        
        print("--- MLX Initialized ---")
        print("Array A:\n\(a)")
        print("Array B:\n\(b)")
        print("Result A + B:\n\(c)")
        print("-----------------------")
    }
}
