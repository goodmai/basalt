import Foundation
import SQLite
import JWTKit
import Crypto
import HummingbirdBcrypt

/// Легковесная база данных (SQLite) и сервис аутентификации (JWT)
public actor AuthService {
    private let db: Connection
    private let signers: JWTSigners
    
    // Таблицы
    private static let users = Table("users")
    private static let id = Expression<Int64>("id")
    private static let username = Expression<String>("username")
    private static let password = Expression<String>("password") // Hashed
    
    private static let blocklist = Table("blocklist")
    private static let tokenCol = Expression<String>("token")
    private static let expiresAt = Expression<Date>("expiresAt")
    
    public init(dbPath: String, jwtSecret: String?) throws {
        let connection = try Connection(dbPath)
        
        // Создание таблиц
        try connection.run(Self.users.create(ifNotExists: true) { t in
            t.column(Self.id, primaryKey: true)
            t.column(Self.username, unique: true)
            t.column(Self.password)
        })
        
        try connection.run(Self.blocklist.create(ifNotExists: true) { t in
            t.column(Self.tokenCol, primaryKey: true)
            t.column(Self.expiresAt)
        })
        
        // SECURITY: No default user created automatically
        // Users must be created via CLI or environment setup
        // if try connection.scalar(Self.users.count) == 0 {
        //     let hashed = Self.hashPassword("admin")
        //     try connection.run(Self.users.insert(Self.username <- "admin", Self.password <- hashed))
        // }
        
        self.db = connection
        
        // Инициализация JWT
        let s = JWTSigners()
        
        // SECURITY: JWT secret must be provided or read from environment
        let secret = jwtSecret ?? ProcessInfo.processInfo.environment["JWT_SECRET"]
        guard let jwtSecretKey = secret, !jwtSecretKey.isEmpty else {
            throw GemError.invalidRequestStructure(
                details: "JWT_SECRET must be provided via environment variable or parameter"
            )
        }
        
        // Warn if secret is too short
        if jwtSecretKey.count < 32 {
            print("⚠️  WARNING: JWT secret is shorter than recommended 32 characters")
        }
        
        s.use(.hs256(key: jwtSecretKey))
        self.signers = s
    }
    
    public struct SessionToken: JWTPayload, Equatable {
        public let sub: String
        public let exp: ExpirationClaim
        
        public func verify(using signer: JWTSigner) throws {
            try exp.verifyNotExpired()
        }
    }
    
    // MARK: — Login
    
    public func login(user: String, pass: String) throws -> String {
        let query = Self.users.filter(Self.username == user)
        guard let row = try db.pluck(query) else {
            throw GemError.invalidRequestStructure(details: "Invalid username or password")
        }
        
        let storedHash = try row.get(Self.password)
        guard Self.verifyPassword(pass, hash: storedHash) else {
            throw GemError.invalidRequestStructure(details: "Invalid username or password")
        }
        
        let payload = SessionToken(
            sub: user,
            exp: ExpirationClaim(value: Date().addingTimeInterval(24 * 3600)) // 24 часа
        )
        
        // Опциональная миграция старых хешей SHA256 на Bcrypt
        if Self.isOldSha256Hash(storedHash) {
            let newHash = Self.hashPassword(pass)
            try db.run(Self.users.filter(Self.username == user).update(Self.password <- newHash))
        }
        
        return try signers.sign(payload)
    }
    
    // MARK: — Private Helpers
    
    private static func isOldSha256Hash(_ hash: String) -> Bool {
        return !hash.hasPrefix("$2") // Bcrypt хеши всегда начинаются с $2a$, $2b$ и т.д.
    }
    
    private static func hashPassword(_ pass: String) -> String {
        return Bcrypt.hash(pass, cost: 12)
    }
    
    private static func verifyPassword(_ pass: String, hash: String) -> Bool {
        // SECURITY: Legacy SHA256 hashes no longer supported
        // All passwords must use Bcrypt. Users with old hashes must reset passwords.
        // if !hash.contains(":") && !hash.hasPrefix("$2") {
        //     // This was insecure - static salt is vulnerable
        //     return false
        // }
        
        // Миграция старых солёных SHA256 хешей
        if hash.contains(":") && !hash.hasPrefix("$2") {
            let parts = hash.split(separator: ":")
            guard parts.count == 2, 
                  let saltData = Data(base64Encoded: String(parts[0])),
                  let salt = String(data: saltData, encoding: .utf8) else {
                return false
            }
            
            let input = pass + salt
            let digest = SHA256.hash(data: Data(input.utf8))
            let hashHex = digest.compactMap { String(format: "%02x", $0) }.joined()
            let expectedHash = "\(String(parts[0])):\(hashHex)"
            
            return expectedHash == hash
        }
        
        // Bcrypt верификация
        return Bcrypt.verify(pass, hash: hash)
    }
    
    // MARK: — Logout
    
    public func logout(token: String) throws {
        let payload = try signers.verify(token, as: SessionToken.self)
        try db.run(Self.blocklist.insert(or: .replace, Self.tokenCol <- token, Self.expiresAt <- payload.exp.value))
        
        // Периодическая очистка старых токенов
        try db.run(Self.blocklist.filter(Self.expiresAt < Date()).delete())
    }
    
    // MARK: — Verify
    
    public func verify(token: String) throws -> String {
        let payload = try signers.verify(token, as: SessionToken.self)
        
        // Проверка в блок-листе
        if try db.scalar(Self.blocklist.filter(Self.tokenCol == token).count) > 0 {
            throw GemError.invalidRequestStructure(details: "Token has been revoked")
        }
        
        return payload.sub
    }
    
    // MARK: — User Management
    
    /// Create a new user with hashed password (Bcrypt)
    /// - Parameters:
    ///   - username: Username (must be unique)
    ///   - password: Plain text password (will be hashed with Bcrypt)
    /// - Throws: Error if username already exists or database error
    public func createUser(username: String, password: String) throws {
        // Validation
        guard !username.isEmpty, username.count <= 100 else {
            throw GemError.invalidRequestStructure(details: "Invalid username")
        }
        
        guard !password.isEmpty, password.count >= 8 else {
            throw GemError.invalidRequestStructure(details: "Password must be at least 8 characters")
        }
        
        // Check if user exists
        let query = Self.users.filter(Self.username == username)
        if try db.pluck(query) != nil {
            throw GemError.invalidRequestStructure(details: "Username already exists")
        }
        
        // Hash password with Bcrypt
        let hashedPassword = Self.hashPassword(password)
        
        // Insert user
        try db.run(Self.users.insert(Self.username <- username, Self.password <- hashedPassword))
    }
    
    /// Check if any users exist in the database
    public func hasUsers() throws -> Bool {
        return try db.scalar(Self.users.count) > 0
    }
}
