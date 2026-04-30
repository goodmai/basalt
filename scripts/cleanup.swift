#!/usr/bin/env swift
import Foundation

let fileManager = FileManager.default
let currentDir = URL(fileURLWithPath: fileManager.currentDirectoryPath)

let directoriesToClean = [
    currentDir.appendingPathComponent("logs"),
    currentDir.appendingPathComponent("images"),
    currentDir.appendingPathComponent("reports")
]

let oneDayAgo = Date().addingTimeInterval(-86400) // 24 hours ago

print("Starting cleanup of old files...")

for dir in directoriesToClean {
    guard fileManager.fileExists(atPath: dir.path) else {
        print("Directory \(dir.path) does not exist, skipping.")
        continue
    }
    
    do {
        let contents = try fileManager.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.creationDateKey, .contentModificationDateKey], options: [.skipsHiddenFiles])
        
        for file in contents {
            let attributes = try file.resourceValues(forKeys: [.creationDateKey, .contentModificationDateKey])
            
            // Checking modification date first, fallback to creation
            let fileDate = attributes.contentModificationDate ?? attributes.creationDate ?? Date()
            
            if fileDate < oneDayAgo {
                try fileManager.removeItem(at: file)
                print("🗑️ Deleted old file: \(file.lastPathComponent) (from \(dir.lastPathComponent))")
            } else {
                print("✅ Kept recent file: \(file.lastPathComponent)")
            }
        }
    } catch {
        print("❌ Error accessing directory \(dir.path): \(error.localizedDescription)")
    }
}

print("Cleanup complete.")
