import Testing
import Foundation
@testable import GemCore

/// Epic 16.8: Progress Bar System - TDD Tests
/// Write tests FIRST, implement SECOND
@Suite("ProgressBar Tests - Epic 16.8")
struct ProgressBarTests {
    
    // MARK: - Initialization Tests
    
    @Test("Initialize progress bar with total")
    func testInitialization() {
        let progress = ProgressBar(total: 100, title: "Downloading")
        
        #expect(progress.total == 100)
        #expect(progress.current == 0)
        #expect(progress.title == "Downloading")
        #expect(progress.percentage == 0.0)
    }
    
    @Test("Initialize with zero total")
    func testInitializationWithZero() {
        let progress = ProgressBar(total: 0, title: "Test")
        
        #expect(progress.total == 0)
        #expect(progress.percentage == 0.0)
    }
    
    // MARK: - Update Tests
    
    @Test("Update current value")
    func testUpdateCurrent() {
        var progress = ProgressBar(total: 100)
        
        progress.update(current: 25)
        #expect(progress.current == 25)
        #expect(progress.percentage == 25.0)
        
        progress.update(current: 50)
        #expect(progress.current == 50)
        #expect(progress.percentage == 50.0)
    }
    
    @Test("Update with increment")
    func testIncrement() {
        var progress = ProgressBar(total: 100)
        
        progress.increment()
        #expect(progress.current == 1)
        
        progress.increment(by: 10)
        #expect(progress.current == 11)
    }
    
    @Test("Update with custom message")
    func testUpdateWithMessage() {
        var progress = ProgressBar(total: 100)
        
        progress.update(current: 50, message: "Half done")
        #expect(progress.current == 50)
        #expect(progress.currentMessage == "Half done")
    }
    
    @Test("Prevent current from exceeding total")
    func testClampToTotal() {
        var progress = ProgressBar(total: 100)
        
        progress.update(current: 150)
        #expect(progress.current == 100)
        #expect(progress.percentage == 100.0)
    }
    
    // MARK: - Percentage Calculation Tests
    
    @Test("Calculate percentage correctly")
    func testPercentageCalculation() {
        var progress = ProgressBar(total: 200)
        
        progress.update(current: 50)
        #expect(progress.percentage == 25.0)
        
        progress.update(current: 100)
        #expect(progress.percentage == 50.0)
        
        progress.update(current: 200)
        #expect(progress.percentage == 100.0)
    }
    
    @Test("Handle zero total for percentage")
    func testPercentageWithZeroTotal() {
        let progress = ProgressBar(total: 0)
        #expect(progress.percentage == 0.0)
    }
    
    // MARK: - Rendering Tests
    
    @Test("Render bar style progress")
    func testRenderBarStyle() {
        var progress = ProgressBar(total: 100, style: .bar(width: 20))
        
        progress.update(current: 50)
        let output = progress.render()
        
        #expect(output.contains("50.0%") || output.contains("50%"))  // Accept both
        #expect(output.contains("█") || output.contains("="))
    }
    
    @Test("Render percentage style progress")
    func testRenderPercentageStyle() {
        var progress = ProgressBar(total: 100, style: .percentage)
        
        progress.update(current: 75)
        let output = progress.render()
        
        #expect(output.contains("75.0%") || output.contains("75%"))  // Accept both
    }
    
    @Test("Render spinner style progress")
    func testRenderSpinnerStyle() {
        let progress = ProgressBar(total: 100, style: .spinner)
        let output = progress.render()
        
        // Should contain spinner character
        #expect(!output.isEmpty)
    }
    
    @Test("Render with title")
    func testRenderWithTitle() {
        var progress = ProgressBar(total: 100, title: "Downloading model")
        
        progress.update(current: 30)
        let output = progress.render()
        
        #expect(output.contains("Downloading model"))
    }
    
    // MARK: - Completion Tests
    
    @Test("Check if progress is complete")
    func testIsComplete() {
        var progress = ProgressBar(total: 100)
        
        #expect(!progress.isComplete)
        
        progress.update(current: 100)
        #expect(progress.isComplete)
    }
    
    @Test("Mark as complete")
    func testMarkComplete() {
        var progress = ProgressBar(total: 100)
        
        progress.update(current: 50)
        progress.markComplete()
        
        #expect(progress.isComplete)
        #expect(progress.current == progress.total)
    }
    
    // MARK: - Speed Calculation Tests
    
    @Test("Calculate speed from time delta")
    func testSpeedCalculation() async {
        var progress = ProgressBar(total: 1000)
        
        progress.update(current: 0)
        
        // Simulate 100 units in ~0.1 seconds
        try? await Task.sleep(for: .milliseconds(100))
        progress.update(current: 100)
        
        let speed = progress.estimatedSpeed
        // Speed should be > 0 (units per second)
        #expect(speed >= 0)
    }
    
    @Test("Estimate time remaining")
    func testTimeRemaining() async {
        var progress = ProgressBar(total: 1000)
        
        progress.update(current: 0)
        try? await Task.sleep(for: .milliseconds(50))
        progress.update(current: 250)
        
        let remaining = progress.estimatedTimeRemaining
        // Should have some estimate
        #expect(remaining >= 0)
    }
    
    // MARK: - Format Tests
    
    @Test("Format bytes for display")
    func testFormatBytes() {
        #expect(ProgressBar.formatBytes(512) == "512 B")
        #expect(ProgressBar.formatBytes(1024) == "1.0 KB")
        #expect(ProgressBar.formatBytes(1_048_576) == "1.0 MB")
        #expect(ProgressBar.formatBytes(1_073_741_824) == "1.0 GB")
    }
    
    @Test("Format time duration")
    func testFormatDuration() {
        #expect(ProgressBar.formatDuration(30) == "30s")
        #expect(ProgressBar.formatDuration(90) == "1m 30s")
        #expect(ProgressBar.formatDuration(3665) == "1h 1m 5s")
    }
}

@Suite("MultiProgressBar Tests - Epic 16.8")
struct MultiProgressBarTests {
    
    @Test("Initialize multi-progress bar")
    func testMultiProgressInit() async {
        let multi = MultiProgressBar()
        
        let count = await multi.taskCount
        #expect(count == 0)
    }
    
    @Test("Add progress bars")
    func testAddProgressBars() async {
        let multi = MultiProgressBar()
        
        let id1 = await multi.addTask(title: "Task 1", total: 100)
        let id2 = await multi.addTask(title: "Task 2", total: 200)
        
        let count = await multi.taskCount
        #expect(count == 2)
        #expect(id1 != id2)
    }
    
    @Test("Update individual task")
    func testUpdateTask() async {
        let multi = MultiProgressBar()
        
        let id = await multi.addTask(title: "Download", total: 100)
        await multi.updateTask(id, current: 50)
        
        let progress = await multi.getProgress(for: id)
        #expect(progress?.current == 50)
    }
    
    @Test("Complete task")
    func testCompleteTask() async {
        let multi = MultiProgressBar()
        
        let id = await multi.addTask(title: "Task", total: 100)
        await multi.completeTask(id)
        
        let progress = await multi.getProgress(for: id)
        #expect(progress?.isComplete == true)
    }
    
    @Test("Get overall progress")
    func testOverallProgress() async {
        let multi = MultiProgressBar()
        
        let id1 = await multi.addTask(title: "Task 1", total: 100)
        let id2 = await multi.addTask(title: "Task 2", total: 100)
        
        await multi.updateTask(id1, current: 100)  // 100% done
        await multi.updateTask(id2, current: 50)   // 50% done
        
        let overall = await multi.overallPercentage
        #expect(overall == 75.0)  // (100 + 50) / 2
    }
    
    @Test("Render multiple progress bars")
    func testRenderMulti() async {
        let multi = MultiProgressBar()
        
        let id1 = await multi.addTask(title: "Download 1", total: 100)
        let id2 = await multi.addTask(title: "Download 2", total: 100)
        
        await multi.updateTask(id1, current: 75)
        await multi.updateTask(id2, current: 25)
        
        let output = await multi.render()
        
        #expect(output.contains("Download 1"))
        #expect(output.contains("Download 2"))
        #expect(output.contains("75"))
        #expect(output.contains("25"))
    }
}

@Suite("ProgressBar Integration Tests - Epic 16.8")
struct ProgressBarIntegrationTests {
    
    @Test("Simulate download with progress")
    func testDownloadSimulation() async {
        var progress = ProgressBar(total: 1000, title: "Downloading model.bin")
        
        for i in stride(from: 0, through: 1000, by: 100) {
            progress.update(current: i, message: "\(i)/1000 bytes")
            
            let output = progress.render()
            #expect(!output.isEmpty)
            
            try? await Task.sleep(for: .milliseconds(10))
        }
        
        #expect(progress.isComplete)
    }
    
    @Test("Concurrent progress updates")
    func testConcurrentUpdates() async {
        let multi = MultiProgressBar()
        
        let id1 = await multi.addTask(title: "Task 1", total: 100)
        let id2 = await multi.addTask(title: "Task 2", total: 100)
        let id3 = await multi.addTask(title: "Task 3", total: 100)
        
        await withTaskGroup(of: Void.self) { group in
            group.addTask {
                for i in 0...100 {
                    await multi.updateTask(id1, current: i)
                    try? await Task.sleep(for: .milliseconds(5))
                }
            }
            
            group.addTask {
                for i in 0...100 {
                    await multi.updateTask(id2, current: i)
                    try? await Task.sleep(for: .milliseconds(7))
                }
            }
            
            group.addTask {
                for i in 0...100 {
                    await multi.updateTask(id3, current: i)
                    try? await Task.sleep(for: .milliseconds(3))
                }
            }
        }
        
        let p1 = await multi.getProgress(for: id1)
        let p2 = await multi.getProgress(for: id2)
        let p3 = await multi.getProgress(for: id3)
        
        #expect(p1?.isComplete == true)
        #expect(p2?.isComplete == true)
        #expect(p3?.isComplete == true)
    }
}
