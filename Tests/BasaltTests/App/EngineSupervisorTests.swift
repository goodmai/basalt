import Foundation
import Synchronization
import Testing
@testable import BasaltAppCore
@testable import BasaltCore

@Suite("Mac app engine supervisor: spawns the helper, reads its ready line, restarts it with backoff, gives up after three crashes, adopts a running one (#434 phase 1)", .timeLimit(.minutes(1)))
struct EngineSupervisorTests {
    final class Fixture: Sendable {
        let folder: URL
        let tick = Mutex(0)

        init() throws {
            folder = FileManager.default.temporaryDirectory.appending(path: "supervisor-\(UUID().uuidString)")
            try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        }

        deinit { try? FileManager.default.removeItem(at: folder) }

        func script(_ body: String) throws -> HelperLaunch {
            let url = folder.appending(path: "helper.sh")
            try Data("#!/bin/sh\n\(body)\n".utf8).write(to: url)
            try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: url.path)
            return HelperLaunch(executable: url, arguments: ["serve", "--model", "org/m"])
        }

        var recordURL: URL { folder.appending(path: "engine.json") }

        func environment(probe: @escaping @Sendable (EngineEndpoint) async -> Bool = { _ in true }) -> EngineEnvironment {
            EngineEnvironment(
                now: { [self] in .seconds(self.tick.withLock { value in value += 1; return value }) },
                sleep: { _ in await Task.yield() }, probe: probe, records: .file(recordURL))
        }

        func file(_ name: String) -> String? { try? String(contentsOf: folder.appending(path: name), encoding: .utf8) }
    }

    static func until(_ supervisor: EngineSupervisor, _ predicate: (EngineState.Phase) -> Bool) async -> EngineState.Phase {
        let current = await supervisor.state.phase
        if predicate(current) { return current }
        for await state in supervisor.states where predicate(state.phase) { return state.phase }
        return await supervisor.state.phase
    }

    static let ready = #"echo "$BASALT_API_KEY" > "$(dirname "$0")/token"; echo "$@" > "$(dirname "$0")/args"; echo 'basalt-ready {"port":4242,"pid":1,"version":"x"}'; exec sleep 600"#

    @Test("The helper gets the token by environment and the port flags; its ready line moves the state to ready on that port, with a 0600 record; stop removes it")
    func readyAndStop() async throws {
        let fixture = try Fixture()
        let supervisor = EngineSupervisor(launch: try fixture.script(Self.ready), environment: fixture.environment())
        await supervisor.start()
        let phase = await Self.until(supervisor) { $0 == .ready(port: 4242) }
        let token = await supervisor.token
        #expect(phase == .ready(port: 4242))
        #expect(fixture.file("token")?.trimmingCharacters(in: .newlines) == token)
        #expect(fixture.file("args")?.contains("serve --model org/m --host 127.0.0.1 --port 0 --announce-ready") == true)
        let record = try #require(EngineRecords.file(fixture.recordURL).load())
        #expect(record.port == 4242 && record.token == token)
        let mode = try FileManager.default.attributesOfItem(atPath: fixture.recordURL.path)[.posixPermissions] as? Int
        #expect(mode == 0o600)
        let endpoint = await supervisor.endpoint
        #expect(endpoint == .loopback(port: 4242, token: token))
        await supervisor.shutdown()
        let stopped = await supervisor.state.phase
        #expect(stopped == .stopped && EngineRecords.file(fixture.recordURL).load() == nil)
        #expect(kill(record.pid, 0) != 0, "the helper outlived shutdown")
    }

    @Test("A helper that exits at once is restarted twice with backoff, then the supervisor gives up with its stderr lines")
    func givesUp() async throws {
        let fixture = try Fixture()
        let supervisor = EngineSupervisor(launch: try fixture.script("echo boom >&2; exit 3"), environment: fixture.environment())
        await supervisor.start()
        let phase = await Self.until(supervisor) { if case .failed = $0 { true } else { false } }
        guard case .failed(let log) = phase else {
            Issue.record("expected failed, got \(phase)")
            return
        }
        let epoch = await supervisor.state.epoch
        #expect(log.contains("boom") && epoch == 3)
    }

    @Test("A helper that crashes once comes back on the next spawn")
    func recovers() async throws {
        let fixture = try Fixture()
        let body = #"cd "$(dirname "$0")"; if [ ! -f crashed ]; then touch crashed; exit 1; fi; "# + Self.ready
        let supervisor = EngineSupervisor(launch: try fixture.script(body), environment: fixture.environment())
        await supervisor.start()
        let phase = await Self.until(supervisor) { $0 == .ready(port: 4242) }
        let epoch = await supervisor.state.epoch
        #expect(phase == .ready(port: 4242) && epoch == 2)
        await supervisor.stop()
    }

    @Test("A live helper in the record with a token that the probe accepts is adopted without spawning; a stale record is ignored")
    func adopts() async throws {
        let fixture = try Fixture()
        let pid = ProcessInfo.processInfo.processIdentifier
        EngineRecords.file(fixture.recordURL).save(EngineRecord(pid: pid, port: 5151, token: "kept"))
        let marker = #"touch "$(dirname "$0")/spawned"; exit 1"#
        let supervisor = EngineSupervisor(launch: try fixture.script(marker), environment: fixture.environment { $0.token == "kept" })
        await supervisor.start()
        let (phase, token) = (await supervisor.state.phase, await supervisor.token)
        #expect(phase == .ready(port: 5151) && token == "kept" && fixture.file("spawned") == nil)
        EngineRecords.file(fixture.recordURL).save(EngineRecord(pid: pid, port: 5151, token: "kept", version: "0.0.1"))
        let stale = EngineSupervisor(launch: try fixture.script(Self.ready), environment: fixture.environment())
        await stale.start()
        let fresh = await Self.until(stale) { $0 == .ready(port: 4242) }
        #expect(fresh == .ready(port: 4242))
        await stale.stop()
    }
}
