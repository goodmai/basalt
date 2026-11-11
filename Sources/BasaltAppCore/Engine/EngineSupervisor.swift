import BasaltCore
import Foundation

public struct HelperLaunch: Sendable, Equatable {
    public var executable: URL
    public var arguments: [String]

    public init(executable: URL, arguments: [String]) { (self.executable, self.arguments) = (executable, arguments) }

    public static let readyPrefix = "basalt-ready "
    public static let logLines = 200
    public static let stopGrace = Duration.seconds(10)
    public static let probeInterval = Duration.milliseconds(250)
}

public actor EngineSupervisor {
    public private(set) var state = EngineState()
    public private(set) var token: String
    public private(set) var log: [String] = []
    public nonisolated let states: AsyncStream<EngineState>
    private let updates: AsyncStream<EngineState>.Continuation
    private let launch: HelperLaunch
    private let environment: EngineEnvironment
    private var process: Process?
    private var adoptedPID: Int32?

    public init(launch: HelperLaunch, environment: EngineEnvironment, token: String = EngineSupervisor.newToken()) {
        (self.launch, self.environment, self.token) = (launch, environment, token)
        (states, updates) = AsyncStream.makeStream()
    }

    public static func newToken() -> String {
        var generator = SystemRandomNumberGenerator()
        return (0 ..< 4).map { _ in String(generator.next() as UInt64, radix: 16) }.joined()
    }

    public var endpoint: EngineEndpoint? { state.port.map { .loopback(port: $0, token: token) } }

    public func start() async {
        if let record = environment.records.load(), record.version == HealthResponse.version, environment.isAlive(record.pid),
           await environment.probe(.loopback(port: record.port, token: record.token)) {
            (token, adoptedPID) = (record.token, record.pid)
            send(.adopted(port: record.port))
            return
        }
        send(.start)
    }

    public func stop() { send(.stop) }

    func send(_ event: EngineEvent) {
        for effect in EngineReducer.reduce(&state, event) { perform(effect) }
        updates.yield(state)
    }

    func perform(_ effect: EngineEffect) {
        switch effect {
        case .spawn(let epoch): spawn(epoch)
        case .kill: terminate()
        case .probe(let epoch, let port): probe(epoch, .loopback(port: port, token: token))
        case .schedule(let epoch, let delay):
            let sleep = environment.sleep
            Task { [weak self] in
                guard (try? await sleep(delay)) != nil else { return }
                await self?.send(.retry(epoch: epoch))
            }
        case .writeRecord(let port):
            if let pid = process?.processIdentifier { environment.records.save(EngineRecord(pid: pid, port: port, token: token)) }
        case .removeRecord: environment.records.remove()
        }
    }

    func spawn(_ epoch: Int) {
        let process = Process()
        process.executableURL = launch.executable
        process.arguments = launch.arguments + ["--host", "127.0.0.1", "--port", "0", "--announce-ready"]
        process.environment = ProcessInfo.processInfo.environment.merging(["BASALT_API_KEY": token]) { $1 }
        let (output, errors) = (Pipe(), Pipe())
        (process.standardOutput, process.standardError) = (output, errors)
        process.terminationHandler = { [weak self] _ in
            Task { await self?.exited(epoch) }
        }
        do {
            try process.run()
        } catch {
            record(["could not start \(launch.executable.path): \(error.localizedDescription)"])
            send(.exited(epoch: epoch, at: environment.now(), log: log))
            return
        }
        self.process = process
        read(output.fileHandleForReading, epoch: epoch, announces: true)
        read(errors.fileHandleForReading, epoch: epoch, announces: false)
    }

    func read(_ handle: FileHandle, epoch: Int, announces: Bool) {
        let chunks = Self.chunks(handle)
        Task { [weak self] in
            var pending = Data()
            for await chunk in chunks {
                pending.append(chunk)
                while let end = pending.firstIndex(of: UInt8(ascii: "\n")) {
                    let text = String(decoding: pending[pending.startIndex ..< end], as: UTF8.self)
                    pending.removeSubrange(pending.startIndex ... end)
                    await self?.line(text, epoch: epoch, announces: announces)
                }
            }
        }
    }

    static func chunks(_ handle: FileHandle) -> AsyncStream<Data> {
        AsyncStream { continuation in
            handle.readabilityHandler = { readable in
                let data = readable.availableData
                guard !data.isEmpty else {
                    readable.readabilityHandler = nil
                    continuation.finish()
                    return
                }
                continuation.yield(data)
            }
            continuation.onTermination = { _ in handle.readabilityHandler = nil }
        }
    }

    func line(_ text: String, epoch: Int, announces: Bool) {
        guard announces, text.hasPrefix(HelperLaunch.readyPrefix) else { return record([text]) }
        let json = (try? JSONSerialization.jsonObject(with: Data(text.dropFirst(HelperLaunch.readyPrefix.count).utf8))) as? [String: Any]
        guard let port = json?["port"] as? Int else { return record([text]) }
        send(.handshake(epoch: epoch, port: port))
    }

    func record(_ lines: [String]) {
        log = Array((log + lines).suffix(HelperLaunch.logLines))
    }

    func exited(_ epoch: Int) {
        if epoch == state.epoch { process = nil }
        send(.exited(epoch: epoch, at: environment.now(), log: log))
    }

    func probe(_ epoch: Int, _ endpoint: EngineEndpoint) {
        let (probe, sleep) = (environment.probe, environment.sleep)
        Task { [weak self] in
            while await self?.state.epoch == epoch {
                if await probe(endpoint) {
                    await self?.send(.ready(epoch: epoch))
                    return
                }
                guard (try? await sleep(HelperLaunch.probeInterval)) != nil else { return }
            }
        }
    }

    func terminate() {
        if let pid = adoptedPID {
            kill(pid, SIGINT)
            adoptedPID = nil
        }
        guard let process, process.isRunning else { return }
        self.process = nil
        process.interrupt()
        let (sleep, isAlive) = (environment.sleep, environment.isAlive)
        let pid = process.processIdentifier
        Task {
            guard (try? await sleep(HelperLaunch.stopGrace)) != nil, isAlive(pid) else { return }
            kill(pid, SIGKILL)
        }
    }
}
