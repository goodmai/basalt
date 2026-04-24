import Foundation
import SQLite
import JWTKit

/// Легковесная база данных (SQLite) и сервис аутентификации (JWT)
public actor AuthService {
    private let db: Connection
    private let signers: JWTSigners
    
    // Таблицы
    private static let users = Table("users")
    private static let id = Expression<Int64>("id")
    private static let username = Expression<String>("username")
    private static let password = Expression<String>("password") // Для демо - plain text
    
    private static let blocklist = Table("blocklist")
    private static let tokenCol = Expression<String>("token")
    private static let expiresAt = Expression<Date>("expiresAt")
    
    public init(dbPath: String, jwtSecret: String = "gemma-super-secret-key") throws {
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
        
        // Создание дефолтного пользователя, если база пустая
        if try connection.scalar(Self.users.count) == 0 {
            try connection.run(Self.users.insert(Self.username <- "admin", Self.password <- "admin"))
        }
        
        self.db = connection
        
        // Инициализация JWT
        let s = JWTSigners()
        s.use(.hs256(key: jwtSecret))
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
        let query = Self.users.filter(Self.username == user && Self.password == pass)
        guard try db.scalar(query.count) > 0 else {
            throw GemmaServerError.invalidRequestStructure(details: "Invalid username or password")
        }
        
        let payload = SessionToken(
            sub: user,
            exp: ExpirationClaim(value: Date().addingTimeInterval(24 * 3600)) // 24 часа
        )
        return try signers.sign(payload)
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
            throw GemmaServerError.invalidRequestStructure(details: "Token has been revoked")
        }
        
        return payload.sub
    }
}
