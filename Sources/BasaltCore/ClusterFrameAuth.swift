import CryptoKit
import Darwin
import Foundation
import Security

public enum ClusterFrameAuth: Sendable {
    public static let frameMagic: UInt32 = 0x544C_4342
    public static let frameVersion: UInt16 = 1
    public static let headerByteCount = 28
    public static let macByteCount = 32
    public static let envelopeOverheadBytes = 60
    public static let maxPayloadByteCount = 512 * 1_048_576

    public enum FrameKind: UInt16, Sendable, Equatable, Codable {
        case handshake = 1
        case tensor = 2
        case control = 3
    }

    public struct NodeIdentity: Sendable, Equatable, Codable {
        public let peerId: String
        public let privateKeyRaw: Data
        public let publicKeyRaw: Data

        public init(peerId: String, privateKeyRaw: Data, publicKeyRaw: Data) throws(ClusterError) {
            let trimmed = peerId.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { throw .invalidTopology("peerId must not be empty") }
            guard let priv = try? Curve25519.Signing.PrivateKey(rawRepresentation: privateKeyRaw),
                  priv.publicKey.rawRepresentation == publicKeyRaw else {
                throw .signatureInvalid(peerId: trimmed)
            }
            self.peerId = trimmed
            self.privateKeyRaw = privateKeyRaw
            self.publicKeyRaw = publicKeyRaw
        }

        public static func generate(peerId: String) throws(ClusterError) -> NodeIdentity {
            let trimmed = peerId.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { throw .invalidTopology("peerId must not be empty") }
            let priv = Curve25519.Signing.PrivateKey()
            return try NodeIdentity(
                peerId: trimmed,
                privateKeyRaw: priv.rawRepresentation,
                publicKeyRaw: priv.publicKey.rawRepresentation)
        }

        public func sign(_ message: Data) throws(ClusterError) -> Data {
            guard let priv = try? Curve25519.Signing.PrivateKey(rawRepresentation: privateKeyRaw),
                  let sig = try? priv.signature(for: message) else {
                throw .signatureInvalid(peerId: peerId)
            }
            return sig
        }
    }

    public struct NoncePair: Sendable, Equatable, Codable {
        public let challenger: Data
        public let responder: Data

        public init(challenger: Data, responder: Data) {
            self.challenger = challenger
            self.responder = responder
        }
    }

    public struct HandshakeChallenge: Sendable, Equatable, Codable {
        public let runId: String
        public let challengerId: String
        public let nonce: Data
        public let publicKey: Data
        public let mac: Data

        public init(runId: String, challengerId: String, nonce: Data, publicKey: Data, mac: Data) {
            self.runId = runId
            self.challengerId = challengerId
            self.nonce = nonce
            self.publicKey = publicKey
            self.mac = mac
        }
    }

    public struct HandshakeAccept: Sendable, Equatable, Codable {
        public let runId: String
        public let responderId: String
        public let nonces: NoncePair
        public let publicKey: Data
        public let signature: Data
        public let mac: Data

        public init(
            runId: String,
            responderId: String,
            nonces: NoncePair,
            publicKey: Data,
            signature: Data,
            mac: Data
        ) {
            self.runId = runId
            self.responderId = responderId
            self.nonces = nonces
            self.publicKey = publicKey
            self.signature = signature
            self.mac = mac
        }

        public var challengerNonce: Data { nonces.challenger }
        public var responderNonce: Data { nonces.responder }
    }

    public struct ReplayWindow: Sendable, Equatable {
        public private(set) var highestSequence: UInt64?
        public private(set) var bitmap: UInt64 = 0

        public init() {}

        public mutating func accept(sequence: UInt64) throws(ClusterError) {
            guard let highest = highestSequence else {
                highestSequence = sequence
                bitmap = 1
                return
            }
            if sequence > highest {
                let shift = sequence - highest
                bitmap = shift >= 64 ? 1 : ((bitmap << shift) | 1)
                highestSequence = sequence
                return
            }
            let delta = highest - sequence
            guard delta > 0, delta < 64 else {
                throw .replaySequenceRejected(sequence: sequence)
            }
            let mask = UInt64(1) << delta
            guard (bitmap & mask) == 0 else {
                throw .replaySequenceRejected(sequence: sequence)
            }
            bitmap |= mask
        }
    }

    public struct DecodedFrame: Sendable, Equatable {
        public let kind: FrameKind
        public let epoch: UInt64
        public let sequence: UInt64
        public let payload: Data

        public init(kind: FrameKind, epoch: UInt64, sequence: UInt64, payload: Data) {
            self.kind = kind
            self.epoch = epoch
            self.sequence = sequence
            self.payload = payload
        }
    }
}

extension ClusterFrameAuth {
    public static func validateBindHost(_ rawHost: String) throws(ClusterError) -> String {
        let trimmed = rawHost.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw .invalidEndpoint("bind host must not be empty") }
        let lowered = trimmed.lowercased()
        let unbracketed = lowered.hasPrefix("[") && lowered.hasSuffix("]")
            ? String(lowered.dropFirst().dropLast())
            : lowered
        let wildcards: Set<String> = ["0.0.0.0", "::", "0:0:0:0:0:0:0:0", "*", "0", "0.0.0.0/0", "::/0", "255.255.255.255"]
        if wildcards.contains(unbracketed) || unbracketed.hasPrefix("0.") || isWildcardAddress(unbracketed) {
            throw .wildcardBindRejected(host: trimmed)
        }
        let octets = unbracketed.split(separator: ".", omittingEmptySubsequences: false)
        if octets.count == 4, octets.allSatisfy({ !$0.isEmpty && $0.allSatisfy(\.isNumber) }) {
            for part in octets {
                guard let val = Int(part), (0 ... 255).contains(val) else {
                    throw .invalidEndpoint("invalid IPv4 address '\(trimmed)'")
                }
            }
            return unbracketed
        }
        guard !unbracketed.contains(where: \.isWhitespace), !unbracketed.contains("/") else {
            throw .invalidEndpoint("invalid bind host '\(trimmed)'")
        }
        return unbracketed
    }

    private static func isWildcardAddress(_ host: String) -> Bool {
        var v4 = in_addr()
        if inet_pton(AF_INET, host, &v4) == 1 {
            return v4.s_addr == 0 || v4.s_addr == UInt32.max
        }
        var v6 = in6_addr()
        if inet_pton(AF_INET6, host, &v6) == 1 {
            let bytes = withUnsafeBytes(of: &v6) { Array($0) }
            if bytes.allSatisfy({ $0 == 0 }) { return true }
            let isV4Mapped = bytes[0..<10].allSatisfy({ $0 == 0 }) && bytes[10] == 0xff && bytes[11] == 0xff
            if isV4Mapped && bytes[12..<16].allSatisfy({ $0 == 0 }) { return true }
        }
        return false
    }

    public static func validateSecret(_ secret: String) throws(ClusterError) -> Data {
        let trimmed = secret.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw .emptySecret }
        return Data(trimmed.utf8)
    }

    public static func randomBytes(count: Int = 32) throws(ClusterError) -> Data {
        var bytes = [UInt8](repeating: 0, count: count)
        let status = SecRandomCopyBytes(kSecRandomDefault, count, &bytes)
        guard status == errSecSuccess else {
            throw .malformedFrame("SecRandomCopyBytes failed with status \(status)")
        }
        return Data(bytes)
    }

    public static func makeChallenge(
        runId: String,
        challenger: NodeIdentity,
        secret: String,
        nonce: Data? = nil
    ) throws(ClusterError) -> HandshakeChallenge {
        let psk = try validateSecret(secret)
        let chalNonce: Data = if let nonce { nonce } else { try randomBytes(count: 32) }
        guard chalNonce.count >= 16 else { throw .malformedFrame("challenge nonce too short") }
        let transcript = challengeTranscript(
            runId: runId, challengerId: challenger.peerId, nonce: chalNonce, publicKey: challenger.publicKeyRaw)
        let tag = Data(HMAC<SHA256>.authenticationCode(for: transcript, using: SymmetricKey(data: psk)))
        return HandshakeChallenge(
            runId: runId,
            challengerId: challenger.peerId,
            nonce: chalNonce,
            publicKey: challenger.publicKeyRaw,
            mac: tag)
    }

    public static func acceptChallenge(
        _ challenge: HandshakeChallenge,
        responder: NodeIdentity,
        secret: String,
        responderNonce: Data? = nil
    ) throws(ClusterError) -> (accept: HandshakeAccept, sessionKey: Data) {
        let psk = try validateSecret(secret)
        guard challenge.nonce.count >= 16 else { throw .malformedFrame("challenge nonce too short") }
        let symKey = SymmetricKey(data: psk)
        let cTranscript = challengeTranscript(
            runId: challenge.runId, challengerId: challenge.challengerId,
            nonce: challenge.nonce, publicKey: challenge.publicKey)
        guard HMAC<SHA256>.isValidAuthenticationCode(challenge.mac, authenticating: cTranscript, using: symKey) else {
            throw .macMismatch
        }
        let rNonce: Data = if let responderNonce { responderNonce } else { try randomBytes(count: 32) }
        guard rNonce.count >= 16 else { throw .malformedFrame("responder nonce too short") }
        let fullTranscript = acceptTranscript(
            challenge: challenge, responderId: responder.peerId,
            responderNonce: rNonce, responderKey: responder.publicKeyRaw)
        let sig = try responder.sign(fullTranscript)
        var macInput = fullTranscript
        macInput.append(sig)
        let tag = Data(HMAC<SHA256>.authenticationCode(for: macInput, using: symKey))
        let accept = HandshakeAccept(
            runId: challenge.runId,
            responderId: responder.peerId,
            nonces: NoncePair(challenger: challenge.nonce, responder: rNonce),
            publicKey: responder.publicKeyRaw,
            signature: sig,
            mac: tag)
        let derived = deriveSessionKey(psk: symKey, runId: challenge.runId, cNonce: challenge.nonce, rNonce: rNonce)
        return (accept, derived)
    }

    public static func finalizeHandshake(
        challenge: HandshakeChallenge,
        accept: HandshakeAccept,
        secret: String
    ) throws(ClusterError) -> Data {
        let psk = try validateSecret(secret)
        guard accept.runId == challenge.runId else {
            throw .runMismatch(expected: challenge.runId, actual: accept.runId)
        }
        guard accept.challengerNonce == challenge.nonce else {
            throw .malformedFrame("handshake challenger nonce mismatch")
        }
        guard challenge.nonce.count >= 16, accept.responderNonce.count >= 16 else {
            throw .malformedFrame("handshake nonce too short")
        }
        let symKey = SymmetricKey(data: psk)
        let fullTranscript = acceptTranscript(
            challenge: challenge, responderId: accept.responderId,
            responderNonce: accept.responderNonce, responderKey: accept.publicKey)
        var macInput = fullTranscript
        macInput.append(accept.signature)
        guard HMAC<SHA256>.isValidAuthenticationCode(accept.mac, authenticating: macInput, using: symKey) else {
            throw .macMismatch
        }
        guard let pub = try? Curve25519.Signing.PublicKey(rawRepresentation: accept.publicKey),
              pub.isValidSignature(accept.signature, for: fullTranscript) else {
            throw .signatureInvalid(peerId: accept.responderId)
        }
        return deriveSessionKey(
            psk: symKey, runId: challenge.runId, cNonce: challenge.nonce, rNonce: accept.responderNonce)
    }
}

extension ClusterFrameAuth {
    public static func sealFrame(
        kind: FrameKind,
        epoch: UInt64,
        sequence: UInt64,
        payload: Data,
        sessionKey: Data
    ) throws(ClusterError) -> Data {
        guard !sessionKey.isEmpty else { throw .emptySecret }
        guard payload.count <= maxPayloadByteCount else {
            throw .malformedFrame("payload size \(payload.count) exceeds limit")
        }
        var header = Data(capacity: headerByteCount)
        appendLE(frameMagic, to: &header)
        appendLE(frameVersion, to: &header)
        appendLE(kind.rawValue, to: &header)
        appendLE(epoch, to: &header)
        appendLE(sequence, to: &header)
        appendLE(UInt32(payload.count), to: &header)
        var hmac = HMAC<SHA256>(key: SymmetricKey(data: sessionKey))
        hmac.update(data: header)
        hmac.update(data: payload)
        let tag = Data(hmac.finalize())
        var frame = Data(capacity: envelopeOverheadBytes + payload.count)
        frame.append(header)
        frame.append(tag)
        frame.append(payload)
        return frame
    }

    public static func openFrame(
        _ raw: RawSpan,
        sessionKey: Data,
        replayWindow: inout ReplayWindow
    ) throws(ClusterError) -> DecodedFrame {
        guard !sessionKey.isEmpty else { throw .emptySecret }
        guard raw.byteCount >= envelopeOverheadBytes else {
            throw .truncatedFrame(expected: envelopeOverheadBytes, actual: raw.byteCount)
        }
        let magic: UInt32 = loadLE(raw, at: 0)
        guard magic == frameMagic else { throw .malformedFrame("invalid frame magic") }
        let version: UInt16 = loadLE(raw, at: 4)
        guard version == frameVersion else { throw .malformedFrame("unsupported frame version \(version)") }
        let rawKind: UInt16 = loadLE(raw, at: 6)
        guard let kind = FrameKind(rawValue: rawKind) else {
            throw .malformedFrame("unknown frame kind \(rawKind)")
        }
        let epoch: UInt64 = loadLE(raw, at: 8)
        let sequence: UInt64 = loadLE(raw, at: 16)
        let payloadLen = Int(loadLE(raw, at: 24) as UInt32)
        guard payloadLen <= maxPayloadByteCount else {
            throw .malformedFrame("payload length \(payloadLen) exceeds maximum")
        }
        let expectedTotal = envelopeOverheadBytes + payloadLen
        guard raw.byteCount == expectedTotal else {
            if raw.byteCount < expectedTotal {
                throw .truncatedFrame(expected: expectedTotal, actual: raw.byteCount)
            }
            throw .malformedFrame("trailing bytes in frame (\(raw.byteCount) vs \(expectedTotal))")
        }
        let payload = try verifyAndCopyPayload(raw, payloadLen: payloadLen, sessionKey: sessionKey)
        try replayWindow.accept(sequence: sequence)
        return DecodedFrame(kind: kind, epoch: epoch, sequence: sequence, payload: payload)
    }

    public static func openFrame(
        _ data: Data,
        sessionKey: Data,
        replayWindow: inout ReplayWindow
    ) throws(ClusterError) -> DecodedFrame {
        try openFrame(data.bytes, sessionKey: sessionKey, replayWindow: &replayWindow)
    }

    private static func verifyAndCopyPayload(
        _ raw: RawSpan,
        payloadLen: Int,
        sessionKey: Data
    ) throws(ClusterError) -> Data {
        let verified: Data? = raw.withUnsafeBytes { buf in
            var hmac = HMAC<SHA256>(key: SymmetricKey(data: sessionKey))
            hmac.update(data: UnsafeRawBufferPointer(rebasing: buf[0 ..< headerByteCount]))
            hmac.update(data: UnsafeRawBufferPointer(rebasing: buf[envelopeOverheadBytes ..< (envelopeOverheadBytes + payloadLen)]))
            let computed = hmac.finalize()
            let tagSlice = buf[headerByteCount ..< envelopeOverheadBytes]
            let diff = zip(computed, tagSlice).reduce(UInt8(0)) { $0 | ($1.0 ^ $1.1) }
            guard diff == 0 else { return nil }
            return Data(buf[envelopeOverheadBytes ..< (envelopeOverheadBytes + payloadLen)])
        }
        guard let verified else { throw .macMismatch }
        return verified
    }

    private static func loadLE<T: FixedWidthInteger & BitwiseCopyable>(_ raw: RawSpan, at offset: Int) -> T {
        T(littleEndian: raw.unsafeLoadUnaligned(fromByteOffset: offset, as: T.self))
    }

    private static func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var le = value.littleEndian
        withUnsafeBytes(of: &le) { data.append(contentsOf: $0) }
    }

    private static func challengeTranscript(runId: String, challengerId: String, nonce: Data, publicKey: Data) -> Data {
        var out = Data("basalt-cluster-chal-v1|".utf8)
        out.append(Data(runId.utf8))
        out.append(0)
        out.append(Data(challengerId.utf8))
        out.append(0)
        out.append(nonce)
        out.append(publicKey)
        return out
    }

    private static func acceptTranscript(
        challenge: HandshakeChallenge,
        responderId: String,
        responderNonce: Data,
        responderKey: Data
    ) -> Data {
        var out = Data("basalt-cluster-accept-v1|".utf8)
        out.append(Data(challenge.runId.utf8))
        out.append(0)
        out.append(Data(challenge.challengerId.utf8))
        out.append(0)
        out.append(Data(responderId.utf8))
        out.append(0)
        out.append(challenge.nonce)
        out.append(responderNonce)
        out.append(challenge.publicKey)
        out.append(responderKey)
        return out
    }

    private static func deriveSessionKey(psk: SymmetricKey, runId: String, cNonce: Data, rNonce: Data) -> Data {
        var salt = cNonce
        salt.append(rNonce)
        let derived = HKDF<SHA256>.deriveKey(
            inputKeyMaterial: psk,
            salt: salt,
            info: Data("basalt-cluster-session-v1|\(runId)".utf8),
            outputByteCount: 32)
        return derived.withUnsafeBytes { Data($0) }
    }
}
