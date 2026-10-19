enum ToolTranscript {
    static let imageOmitted = "[image omitted]"

    static func call(name: String, arguments: String) -> String {
        "[Assistant called \(name)]: \(arguments)"
    }

    static func result(_ output: String, for name: String?, isError: Bool = false) -> String {
        "[Tool \(isError ? "error" : "result")\(name.map { " for \($0)" } ?? "")]: \(output)"
    }
}
