import Foundation

struct Cx: Equatable, Sendable {
    var re: Double
    var im: Double

    static let zero = Cx(re: 0, im: 0)
    static let one = Cx(re: 1, im: 0)

    static func polar(_ radius: Double, _ angle: Double) -> Cx { Cx(re: radius * cos(angle), im: radius * sin(angle)) }
    static func + (a: Cx, b: Cx) -> Cx { Cx(re: a.re + b.re, im: a.im + b.im) }
    static func - (a: Cx, b: Cx) -> Cx { Cx(re: a.re - b.re, im: a.im - b.im) }
    static func * (a: Cx, b: Cx) -> Cx { Cx(re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re) }
    static func * (a: Cx, s: Double) -> Cx { Cx(re: a.re * s, im: a.im * s) }
    static func / (a: Cx, b: Cx) -> Cx { (a * b.conjugate) * (1 / b.norm2) }

    var conjugate: Cx { Cx(re: re, im: -im) }
    var norm2: Double { re * re + im * im }
    var magnitude: Double { norm2.squareRoot() }
}

enum KeyTrajectoryFit {
    static let orders = [4, 8, 16]
    static let window = 1000
    static let lowError = 0.1
    static let lowErrorShare = 0.15
    static let bimodalThreshold = 5.0 / 9.0

    struct Fit: Sendable {
        let poles: [Cx]
        let residual: Double
        let energy: Double

        var relative: Double { (residual / energy).squareRoot() }
    }

    struct HeadError: Sendable {
        var relative: [Double]
        var skipped: Int
    }

    static func fit(_ x: [Cx], order: Int) -> Fit? {
        let n = x.count
        let energy = x.reduce(0) { $0 + $1.norm2 }
        guard order > 0, n > 2 * order, energy > 0, energy.isFinite else { return nil }
        let lagged = (1 ... order).map { k in Array(x[(order - k) ..< (n - k)]) }
        guard let coefficients = leastSquares(columns: lagged, target: x[order...].map { $0 * -1 }) else { return nil }
        let poles = roots([Cx.one] + coefficients).map { $0.magnitude > 1 ? $0 * (1 / $0.magnitude) : $0 }
        let basis = poles.map { pole in
            var power = Cx.one
            return (0 ..< n).map { _ in defer { power = power * pole }; return power }
        }
        guard let residues = leastSquares(columns: basis, target: x) else { return nil }
        var residual = 0.0
        for t in 0 ..< n {
            var estimate = Cx.zero
            for m in 0 ..< order { estimate = estimate + basis[m][t] * residues[m] }
            residual += (x[t] - estimate).norm2
        }
        guard residual.isFinite else { return nil }
        return Fit(poles: poles, residual: residual, energy: energy)
    }

    static func leastSquares(columns: [[Cx]], target: [Cx]) -> [Cx]? {
        let m = columns.count
        var gram = [[Cx]](repeating: [Cx](repeating: .zero, count: m), count: m)
        var rhs = [Cx](repeating: .zero, count: m)
        for i in 0 ..< m {
            for j in i ..< m {
                var sum = Cx.zero
                for t in target.indices { sum = sum + columns[i][t].conjugate * columns[j][t] }
                gram[i][j] = sum
                gram[j][i] = sum.conjugate
            }
            for t in target.indices { rhs[i] = rhs[i] + columns[i][t].conjugate * target[t] }
        }
        let ridge = 1e-12 * (gram.indices.map { gram[$0][$0].re }.max() ?? 0) + 1e-300
        for i in 0 ..< m { gram[i][i].re += ridge }
        return choleskySolve(gram, rhs)
    }

    static func choleskySolve(_ a: [[Cx]], _ b: [Cx]) -> [Cx]? {
        let m = b.count
        var lower = [[Cx]](repeating: [Cx](repeating: .zero, count: m), count: m)
        for j in 0 ..< m {
            let diagonal = a[j][j].re - (0 ..< j).reduce(0) { $0 + lower[j][$1].norm2 }
            guard diagonal > 0, diagonal.isFinite else { return nil }
            lower[j][j] = Cx(re: diagonal.squareRoot(), im: 0)
            for i in (j + 1) ..< m {
                let sum = (0 ..< j).reduce(a[i][j]) { $0 - lower[i][$1] * lower[j][$1].conjugate }
                lower[i][j] = sum * (1 / lower[j][j].re)
            }
        }
        var y = [Cx](repeating: .zero, count: m)
        for i in 0 ..< m { y[i] = (0 ..< i).reduce(b[i]) { $0 - lower[i][$1] * y[$1] } * (1 / lower[i][i].re) }
        var x = [Cx](repeating: .zero, count: m)
        for i in stride(from: m - 1, through: 0, by: -1) {
            x[i] = ((i + 1) ..< m).reduce(y[i]) { $0 - lower[$1][i].conjugate * x[$1] } * (1 / lower[i][i].re)
        }
        return x
    }

    static func roots(_ monic: [Cx]) -> [Cx] {
        let degree = monic.count - 1
        let seed = Cx(re: 0.4, im: 0.9)
        var z = [Cx](repeating: .one, count: degree)
        for i in 1 ..< max(degree, 1) { z[i] = z[i - 1] * seed }
        for _ in 0 ..< 1000 {
            var largestStep = 0.0
            for i in 0 ..< degree {
                let value = monic.reduce(Cx.zero) { $0 * z[i] + $1 }
                let spread = (0 ..< degree).reduce(Cx.one) { $1 == i ? $0 : $0 * (z[i] - z[$1]) }
                let step = spread.norm2 > 0 ? value / spread : Cx(re: 1e-9, im: 1e-9)
                z[i] = z[i] - step
                largestStep = max(largestStep, step.magnitude)
            }
            if largestStep < 1e-15 { break }
        }
        return z
    }

    static func channelPairs(headDim: Int, ropeHalf: Int) -> [(re: Int, im: Int)] {
        let tail = (headDim - 2 * ropeHalf) / 2
        return (0 ..< ropeHalf).map { ($0, $0 + ropeHalf) }
            + (0 ..< tail).map { (2 * ropeHalf + $0, 2 * ropeHalf + tail + $0) }
    }

    static func headError(_ keys: ArraySlice<Float>, tokens: Int, headDim: Int, ropeHalf: Int) -> HeadError {
        let start = keys.startIndex
        var residual = [Double](repeating: 0, count: orders.count)
        var energy = [Double](repeating: 0, count: orders.count)
        var skipped = 0
        for first in stride(from: 0, through: tokens - window, by: window) {
            for pair in channelPairs(headDim: headDim, ropeHalf: ropeHalf) {
                let x = (first ..< first + window).map { t in
                    Cx(re: Double(keys[start + t * headDim + pair.re]), im: Double(keys[start + t * headDim + pair.im]))
                }
                for (index, order) in orders.enumerated() {
                    guard let fit = fit(x, order: order) else { skipped += 1; continue }
                    residual[index] += fit.residual
                    energy[index] += fit.energy
                }
            }
        }
        return HeadError(relative: zip(residual, energy).map { $1 > 0 ? ($0 / $1).squareRoot() : .nan }, skipped: skipped)
    }

    static func bimodality(_ values: [Double]) -> Double {
        let n = Double(values.count)
        guard values.count > 3 else { return 0 }
        let mean = values.reduce(0, +) / n
        let moment = { (power: Double) in values.reduce(0) { $0 + pow($1 - mean, power) } / n }
        let variance = moment(2)
        guard variance > 0 else { return 0 }
        let skew = moment(3) / pow(variance, 1.5) * (n * (n - 1)).squareRoot() / (n - 2)
        let excess = ((n + 1) * (moment(4) / (variance * variance) - 3) + 6) * (n - 1) / ((n - 2) * (n - 3))
        return (skew * skew + 1) / (excess + 3 * (n - 1) * (n - 1) / ((n - 2) * (n - 3)))
    }

    static func phaseOneGo(_ errors: [Double]) -> (lowShare: Double, bimodality: Double, go: Bool) {
        let finite = errors.filter(\.isFinite)
        let share = finite.isEmpty ? 0 : Double(finite.filter { $0 < lowError }.count) / Double(finite.count)
        let coefficient = bimodality(finite)
        return (share, coefficient, share >= lowErrorShare && coefficient > bimodalThreshold)
    }
}
