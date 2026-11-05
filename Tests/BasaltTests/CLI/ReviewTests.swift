import Testing
import Foundation
@testable import BasaltCore
@testable import BasaltCLI
@testable import BasaltUI

@Suite("basalt review: diff-aware symbol context and self-hosted GitLab / Gitea MR review (#35)")
struct ReviewTests {
    static let multiFileDiff = """
        diff --git a/Sources/Auth.swift b/Sources/Auth.swift
        --- a/Sources/Auth.swift
        +++ b/Sources/Auth.swift
        @@ -10,6 +10,6 @@ public struct Authenticator {
             public func verify(user: String, password: String) async throws -> Bool {
        -        let hash = sha(password)
        +        let hash = bcrypt(password)
                 return store.matches(user, hash)
             }
        --- a/src/lib.rs
        +++ b/src/lib.rs
        @@ -1,3 +1,4 @@
        +pub async fn load(path: &Path) -> Result<Config> {
             todo!()
         }
        --- a/tools/report.py
        +++ b/tools/report.py
        @@ -4,2 +4,3 @@ class Report:
             def render(self, rows):
        +        rows = sorted(rows)
        --- a/web/api.ts
        +++ b/web/api.ts
        @@ -1,2 +1,2 @@
        -export async function fetchUser(id: string) {
        +export async function fetchUser(id: string, signal?: AbortSignal) {
        --- a/README.md
        +++ b/README.md
        @@ -1 +1 @@
        -# def not_code():
        +# class NotCode:
        """

    @Test("Touched signatures are found across Swift, Rust, Python and TypeScript files, enclosing ones from the hunk header first; other files add none")
    func symbolsAcrossLanguages() {
        let symbols = ReviewContext.symbols(in: DiffParser.parseUnifiedDiff(Self.multiFileDiff))
        #expect(symbols == [
            .init(file: "Sources/Auth.swift", signature: "public struct Authenticator"),
            .init(file: "Sources/Auth.swift", signature: "public func verify(user: String, password: String) async throws -> Bool"),
            .init(file: "src/lib.rs", signature: "pub async fn load(path: &Path) -> Result<Config>"),
            .init(file: "tools/report.py", signature: "class Report"),
            .init(file: "tools/report.py", signature: "def render(self, rows)"),
            .init(file: "web/api.ts", signature: "export async function fetchUser(id: string)"),
            .init(file: "web/api.ts", signature: "export async function fetchUser(id: string, signal?: AbortSignal)"),
        ])
    }

    @Test("Under a tight budget the signatures come before any diff hunk, and the dropped hunks are counted")
    func signaturesFirstUnderBudget() throws {
        let full = ReviewContext.prompt(diff: Self.multiFileDiff, budgetChars: 100_000)
        #expect(full.contains("+        let hash = bcrypt(password)"))
        #expect(!full.contains("truncated"))
        let budget = full.components(separatedBy: "## Diff").first!.count + 120
        let tight = ReviewContext.prompt(diff: Self.multiFileDiff, budgetChars: budget)
        #expect(tight.count <= budget)
        #expect(tight.contains("- web/api.ts: `export async function fetchUser(id: string, signal?: AbortSignal)`"))
        #expect(!tight.contains("bcrypt"))
        #expect(tight.contains("[diff truncated: 0 of 5 hunks fit the context budget]"))
        let clipped = ReviewContext.prompt(diff: Self.multiFileDiff, budgetChars: 40)
        #expect(clipped.count == 40)
    }

    @Test("Instance URLs must be https, or http on loopback", arguments: [
        ("https://gitlab.corp.example", true), ("http://127.0.0.1:3000", true), ("http://localhost:3000/gitea", true),
        ("http://gitlab.corp.example", false), ("ftp://gitlab.corp.example", false), ("https://gitlab.corp.example/?x=1", false),
        ("gitlab.corp.example", false),
    ])
    func instanceScheme(instance: String, allowed: Bool) {
        let remote = try? ReviewRemote(forge: .gitlab, instance: instance, repo: "g/p", number: 1, token: nil)
        #expect((remote != nil) == allowed)
    }

    static func stub(status: Int = 200, headers: [String: String] = [:], body: Data) -> String {
        StubURLProtocol.host { _, _ in (status, headers, body) }
    }

    static var session: URLSession { URLSession(configuration: StubURLProtocol.configuration) }

    @Test("GitLab: the MR diffs are fetched with PRIVATE-TOKEN and joined into one unified diff the context builder parses")
    func gitlabFetch() async throws {
        let files = [["old_path": "a.swift", "new_path": "a.swift", "diff": "@@ -1 +1 @@\n-func old() {}\n+func new() {}\n"]]
        let host = Self.stub(body: try JSONSerialization.data(withJSONObject: files))
        let remote = try ReviewRemote(forge: .gitlab, instance: "https://\(host)", repo: "group/sub project", number: 42, token: "glpat-secret")
        let diff = try await remote.fetchDiff(session: Self.session)
        let request = try #require(StubURLProtocol.requests(for: host).first?.0)
        #expect(request.url?.absoluteString == "https://\(host)/api/v4/projects/group%2Fsub%20project/merge_requests/42/diffs?per_page=100")
        #expect(request.value(forHTTPHeaderField: "PRIVATE-TOKEN") == "glpat-secret")
        #expect(ReviewContext.symbols(in: DiffParser.parseUnifiedDiff(diff)).map(\.signature) == ["func old() {}", "func new() {}"])
    }

    @Test("Gitea: the PR .diff is fetched with a token Authorization header")
    func giteaFetch() async throws {
        let host = Self.stub(body: Data("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+def f():\n".utf8))
        let remote = try ReviewRemote(forge: .gitea, instance: "https://\(host)/", repo: "owner/name", number: 7, token: "t0k")
        let diff = try await remote.fetchDiff(session: Self.session)
        let request = try #require(StubURLProtocol.requests(for: host).first?.0)
        #expect(request.url?.absoluteString == "https://\(host)/api/v1/repos/owner/name/pulls/7.diff")
        #expect(request.value(forHTTPHeaderField: "Authorization") == "token t0k")
        #expect(diff.hasSuffix("+def f():\n"))
    }

    @Test("A response over the diff cap is refused")
    func diffCap() async throws {
        let host = Self.stub(body: Data(repeating: 0x2B, count: ReviewRemote.maxDiffBytes + 1))
        let remote = try ReviewRemote(forge: .gitea, instance: "https://\(host)", repo: "o/r", number: 1, token: nil)
        await #expect(throws: ReviewRemote.Failure.self) { try await remote.fetchDiff(session: Self.session) }
    }

    @Test("The request delegate refuses every redirect, so the token header never follows a Location to another host")
    func noRedirect() async throws {
        let target = try #require(URL(string: "https://elsewhere.test/steal"))
        let task = URLSession.shared.dataTask(with: target)
        let response = try #require(HTTPURLResponse(url: target, statusCode: 302, httpVersion: nil, headerFields: ["Location": target.absoluteString]))
        let next = await NoRedirects().urlSession(.shared, task: task, willPerformHTTPRedirection: response, newRequest: URLRequest(url: target))
        #expect(next == nil)
    }

    @Test("An HTTP error names the status, redacts secrets echoed in the body and never includes the token")
    func errorRedaction() async throws {
        let echoed = "bad token glpat-" + String(repeating: "x", count: 24)
        let host = Self.stub(status: 401, body: Data(echoed.utf8))
        let remote = try ReviewRemote(forge: .gitlab, instance: "https://\(host)", repo: "1", number: 3, token: "glpat-" + String(repeating: "x", count: 24))
        let error = await #expect(throws: ReviewRemote.Failure.self) { try await remote.fetchDiff(session: Self.session) }
        #expect(error?.description.contains("HTTP 401") == true)
        #expect(error?.description.contains("[REDACTED:gitlab_token]") == true)
        #expect(error?.description.contains("xxxxxxxx") == false)
    }

    @Test("Posting sends one JSON comment to the MR notes endpoint, with secrets redacted")
    func postComment() async throws {
        let host = Self.stub(status: 201, body: Data("{}".utf8))
        let remote = try ReviewRemote(forge: .gitlab, instance: "https://\(host)", repo: "7", number: 9, token: "glpat-secret")
        try await remote.post("Leaked sk-" + String(repeating: "a", count: 24), session: Self.session)
        let (request, body) = try #require(StubURLProtocol.requests(for: host).first)
        #expect(request.httpMethod == "POST")
        #expect(request.url?.path == "/api/v4/projects/7/merge_requests/9/notes")
        let sent = try #require(body)
        let json = try #require(try JSONSerialization.jsonObject(with: sent) as? [String: String])
        #expect(json["body"] == "Leaked [REDACTED:openai_key]")
    }

    @Test("--print-context fetches the MR and prints the prompt; nothing is posted even with --post")
    func printContextNeverPosts() async throws {
        let host = Self.stub(body: Data("--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n+def f():\n".utf8))
        let command = try ReviewCommand.parse(["--gitea", "https://\(host)", "--repo", "o/r", "--number", "2", "--print-context", "--post"])
        try await command.perform(session: Self.session, environment: ["GITEA_TOKEN": "t"])
        #expect(StubURLProtocol.requests(for: host).map { $0.0.httpMethod ?? "GET" } == ["GET"])
    }

    @Test("Arguments: one source, a remote needs --repo and --number, --post needs a remote, a model or --print-context", arguments: [
        ["--diff", "-", "--print-context"],
        ["--gitlab", "https://g.example", "--repo", "a/b", "--number", "1", "--model", "m", "--post"],
    ])
    func validArguments(arguments: [String]) {
        #expect(throws: Never.self) { try ReviewCommand.parse(arguments) }
    }

    @Test("Invalid argument combinations are rejected", arguments: [
        [String](), ["--diff", "-"], ["--diff", "-", "--gitea", "https://g.example", "--print-context"],
        ["--gitea", "https://g.example", "--print-context"], ["--diff", "-", "--print-context", "--post"],
        ["--diff", "-", "--print-context", "--context-chars", "0"],
    ])
    func invalidArguments(arguments: [String]) {
        #expect(throws: (any Error).self) { try ReviewCommand.parse(arguments) }
    }

    @Test("A plain http instance is rejected before any request")
    func httpRejected() throws {
        let command = try ReviewCommand.parse(["--gitlab", "http://gitlab.corp.example", "--repo", "a", "--number", "1", "--print-context"])
        #expect(throws: (any Error).self) { try command.resolvedRemote(environment: [:]) }
    }
}
