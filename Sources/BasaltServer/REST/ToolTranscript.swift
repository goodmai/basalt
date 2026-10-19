enum ToolTranscript {
    static func call(name: String, arguments: String) -> String {
        "[Assistant called \(name)]: \(arguments)"
    }

    static func result(_ output: String, isError: Bool = false) -> String {
        "[Tool \(isError ? "error" : "result")]: \(output)"
    }
}
