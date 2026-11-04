import Foundation

public struct Tagged<Tag, RawValue>: RawRepresentable {
    public var rawValue: RawValue

    public init(rawValue: RawValue) {
        self.rawValue = rawValue
    }

    public init(_ rawValue: RawValue) {
        self.rawValue = rawValue
    }
}

extension Tagged: Sendable where RawValue: Sendable {}
extension Tagged: Equatable where RawValue: Equatable {}
extension Tagged: Hashable where RawValue: Hashable {}

extension Tagged: Comparable where RawValue: Comparable {
    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rawValue < rhs.rawValue }
}

extension Tagged: ExpressibleByIntegerLiteral where RawValue: ExpressibleByIntegerLiteral {
    public init(integerLiteral value: RawValue.IntegerLiteralType) {
        self.init(RawValue(integerLiteral: value))
    }
}

extension Tagged: AdditiveArithmetic where RawValue: AdditiveArithmetic {
    public static var zero: Self { Self(.zero) }
    public static func + (lhs: Self, rhs: Self) -> Self { Self(lhs.rawValue + rhs.rawValue) }
    public static func - (lhs: Self, rhs: Self) -> Self { Self(lhs.rawValue - rhs.rawValue) }
}

extension Tagged where RawValue: Numeric {
    public static func * (lhs: Self, rhs: RawValue) -> Self { Self(lhs.rawValue * rhs) }
}

extension Tagged where RawValue: BinaryInteger {
    public init(_ value: some BinaryInteger) {
        self.init(rawValue: RawValue(value))
    }

    public static func / (lhs: Self, rhs: RawValue) -> Self { Self(lhs.rawValue / rhs) }
}

extension Tagged: Codable where RawValue: Codable {
    public init(from decoder: any Decoder) throws {
        self.init(try decoder.singleValueContainer().decode(RawValue.self))
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.singleValueContainer()
        try container.encode(rawValue)
    }
}

extension Tagged: CustomStringConvertible {
    public var description: String { "\(rawValue)" }
}

public enum TokenUnit {}
public enum ByteUnit {}

public typealias TokenCount = Tagged<TokenUnit, Int>
public typealias ByteCount = Tagged<ByteUnit, Int64>
