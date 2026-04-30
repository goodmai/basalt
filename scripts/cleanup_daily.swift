#!/usr/bin/swift
import Foundation

let fm = FileManager.default
let currentDir = URL(fileURLWithPath: fm.currentDirectoryPath)

let targetDirs = ["logs", "images", "reports"]

print("🧹 Starting daily cleanup...")

for dirName in targetDirs {
    let dirURL = currentDir.appendingPathComponent(dirName)
    
    var isDir: ObjCBool = false
    if fm.fileExists(atPath: dirURL.path, isDirectory: &isDir), isDir.boolValue {
        do {
            let contents = try fm.contentsOfDirectory(at: dirURL, includingPropertiesForKeys: nil)
            for fileURL in contents {
                // Keep .gitkeep if present
                if fileURL.lastPathComponent == ".gitkeep" { continue }
                
                try fm.removeItem(at: fileURL)
                print("  🗑️ Deleted: \(dirName)/\(fileURL.lastPathComponent)")
            }
        } catch {
            print("  ❌ Failed to clean \(dirName): \(error.localizedDescription)")
        }
    } else {
        print("  ⚠️ Directory \(dirName) does not exist, skipping.")
    }
}

print("✅ Cleanup complete.")
