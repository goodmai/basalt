import Foundation
import UniformTypeIdentifiers
import BasaltCore

enum MCPImages {
    static func parse(_ raw: Any?, allowedIn directories: [URL]) throws(MCPFailure) -> [ImageInput] {
        guard let raw, !(raw is NSNull) else { return [] }
        guard let entries = raw as? [[String: Any]] else {
            throw .invalidParams("images must be an array of {data, mime} or {path} objects")
        }
        do throws(BasaltError) {
            try ImageInput.checkCount(entries.count)
            var images: [ImageInput] = []
            for entry in entries { images.append(try image(entry, allowedIn: directories)) }
            return images
        } catch {
            throw MCPFailure(error)
        }
    }

    static func image(_ entry: [String: Any], allowedIn directories: [URL]) throws(BasaltError) -> ImageInput {
        if let path = entry["path"] as? String { return try file(path, allowedIn: directories) }
        guard let data = entry["data"] as? String, let mime = entry["mime"] as? String else {
            throw .invalidRequestStructure(details: "each image needs data and mime, or a path")
        }
        return try ImageInput(base64: data, mime: mime)
    }

    static func file(_ path: String, allowedIn directories: [URL]) throws(BasaltError) -> ImageInput {
        let target = resolved(URL(fileURLWithPath: path))
        guard directories.contains(where: { contains($0, target) }) else {
            throw .invalidRequestStructure(details: "the image path \(path) is not inside a directory given with serve --image-dir")
        }
        guard let data = try? Data(contentsOf: target, options: .mappedIfSafe) else {
            throw .invalidRequestStructure(details: "the image at \(path) cannot be read")
        }
        return try ImageInput(data: data, mime: UTType(filenameExtension: target.pathExtension)?.preferredMIMEType ?? target.pathExtension)
    }

    static func resolved(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    static func contains(_ directory: URL, _ file: URL) -> Bool {
        let (root, parts) = (directory.pathComponents, file.pathComponents)
        return parts.count > root.count && Array(parts.prefix(root.count)) == root
    }
}
