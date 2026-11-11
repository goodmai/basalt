import Foundation
import Security
import Synchronization

public struct KeychainClient: Sendable {
    public var read: @Sendable (_ account: String) -> String?
    public var write: @Sendable (_ account: String, _ value: String) -> Bool
    public var delete: @Sendable (_ account: String) -> Void

    public init(read: @escaping @Sendable (String) -> String?, write: @escaping @Sendable (String, String) -> Bool, delete: @escaping @Sendable (String) -> Void) {
        (self.read, self.write, self.delete) = (read, write, delete)
    }
}

extension KeychainClient {
    public static let service = "app.basalt.mac"

    public static func live(service: String = service) -> KeychainClient {
        @Sendable func query(_ account: String) -> [String: Any] {
            [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: account]
        }
        return KeychainClient(
            read: { account in
                var item: CFTypeRef?
                let lookup = query(account).merging([kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne]) { $1 }
                guard SecItemCopyMatching(lookup as CFDictionary, &item) == errSecSuccess, let data = item as? Data else { return nil }
                return String(data: data, encoding: .utf8)
            },
            write: { account, value in
                SecItemDelete(query(account) as CFDictionary)
                let item = query(account).merging([kSecValueData as String: Data(value.utf8), kSecAttrAccessible as String: kSecAttrAccessibleWhenUnlocked]) { $1 }
                return SecItemAdd(item as CFDictionary, nil) == errSecSuccess
            },
            delete: { account in SecItemDelete(query(account) as CFDictionary) })
    }

    public static func memory() -> KeychainClient {
        let items = Mutex<[String: String]>([:])
        return KeychainClient(
            read: { account in items.withLock { $0[account] } },
            write: { account, value in items.withLock { $0[account] = value }; return true },
            delete: { account in items.withLock { $0[account] = nil } })
    }
}

public enum CloudKeys {
    public static let defaultsKey = "cloudKeyNames"

    public static func isName(_ name: String) -> Bool {
        guard let first = name.unicodeScalars.first, first == "_" || ("A" ... "Z").contains(first) else { return false }
        return name.unicodeScalars.allSatisfy { $0 == "_" || ("A" ... "Z").contains($0) || ("0" ... "9").contains($0) }
    }

    public static func environment(_ names: [String], keychain: KeychainClient) -> [String: String] {
        names.filter(isName).reduce(into: [:]) { environment, name in
            if let value = keychain.read(name), !value.isEmpty { environment[name] = value }
        }
    }
}

public struct DraftStore: Sendable {
    public var load: @Sendable (ConversationID) -> String
    public var save: @Sendable (ConversationID, String) -> Void

    public init(load: @escaping @Sendable (ConversationID) -> String, save: @escaping @Sendable (ConversationID, String) -> Void) {
        (self.load, self.save) = (load, save)
    }

    public static func defaults(_ suite: String? = nil) -> DraftStore {
        let key: @Sendable (ConversationID) -> String = { "draft.\($0.raw.uuidString)" }
        return DraftStore(
            load: { UserDefaults(suiteName: suite)?.string(forKey: key($0)) ?? "" },
            save: { id, text in
                let defaults = UserDefaults(suiteName: suite)
                text.isEmpty ? defaults?.removeObject(forKey: key(id)) : defaults?.set(text, forKey: key(id))
            })
    }
}
