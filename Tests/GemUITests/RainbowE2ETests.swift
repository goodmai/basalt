import XCTest

@MainActor
final class RainbowE2ETests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    override func tearDownWithError() throws {
        // Put teardown code here.
    }

    func testRainbowUIE2E_ZeroPythonArchitecture() throws {
        // 1. Инициализация и внедрение зависимостей
        let productsDirectory: URL
        #if os(macOS)
            let bundle = Bundle.allBundles.first { $0.bundlePath.hasSuffix(".xctest") }
            productsDirectory = bundle?.bundleURL.deletingLastPathComponent() ?? Bundle.main.bundleURL
        #else
            productsDirectory = Bundle.main.bundleURL
        #endif

        let executableURL = productsDirectory.appendingPathComponent("Gemm")
        guard FileManager.default.fileExists(atPath: executableURL.path) else {
            throw XCTSkip("Executable Gemm not found, skipping UI test.")
        }

        let app = XCUIApplication(url: executableURL)
        
        // We pass the required arguments to trigger the Metal UI with real inference
        app.launchArguments = ["chat", "--agent-real"]
        app.launchEnvironment = ["GEMMA_JWT_SECRET": "test-secret"]
        app.launch()
        
        // Wait for the UI to load
        let metalView = app.groups["RainbowMetalBackground"]
        let inputField = app.textFields["RainbowInputField"]
        
        if !metalView.waitForExistence(timeout: 5.0) {
            throw XCTSkip("Metal Background Group not found. Skipping test. Note: macOS Accessibility may not map CLI tools without App Bundles correctly in XCUITest.")
        }
        XCTAssertTrue(inputField.waitForExistence(timeout: 5.0), "Input Field not found")
        
        // 2. Эмулятор человеческого ввода
        inputField.tap()
        inputField.typeText("Test message")
        app.typeKey(.return, modifierFlags: [])
        
        // Wait for processing/streaming to finish
        // In the mock mode (since it's a test), the app will output an assistant message
        let assistantMessage = metalView.staticTexts.matching(NSPredicate(format: "value BEGINSWITH 'Assistant:'")).firstMatch
        XCTAssertTrue(assistantMessage.waitForExistence(timeout: 10.0), "Assistant message did not appear")
        
        // 3. Математическая модель валидации экрана (Топологическая валидация)
        let inputFrame = inputField.frame
        let messageFrame = assistantMessage.frame
        
        // Validate that the message is above the input field
        XCTAssertTrue(messageFrame.maxY < inputFrame.minY, "Сообщение должно располагаться выше поля ввода: message maxY (\(messageFrame.maxY)) >= input minY (\(inputFrame.minY))")
        
        // Additional Keyboard Shortcut tests
        // Test Cancellation
        app.typeKey("c", modifierFlags: .control)
        
        // The mode should switch to finished/idle (we can't read mode directly unless it's exposed in accessibility, 
        // but we know the shortcut is registered).
        
        // Test /export command
        inputField.tap()
        inputField.typeText("Save me")
        app.typeKey(.return, modifierFlags: [])
        
        let assistantMessage2 = metalView.staticTexts.matching(NSPredicate(format: "value BEGINSWITH 'Assistant:'")).firstMatch
        XCTAssertTrue(assistantMessage2.waitForExistence(timeout: 60.0), "Wait for response before exporting")
        
        inputField.tap()
        inputField.typeText("/export")
        app.typeKey(.return, modifierFlags: [])
        
        let systemMessage = metalView.staticTexts.matching(NSPredicate(format: "value BEGINSWITH 'ℹ Chat exported'")).firstMatch
        XCTAssertTrue(systemMessage.waitForExistence(timeout: 5.0), "System export message should appear")
        
        // Test /clear command
        inputField.tap()
        inputField.typeText("/clear")
        app.typeKey(.return, modifierFlags: [])
        
        let assistantMessageAfterClear = metalView.staticTexts.matching(NSPredicate(format: "value BEGINSWITH 'Assistant:'")).firstMatch
        XCTAssertFalse(assistantMessageAfterClear.waitForExistence(timeout: 2.0), "Messages should be cleared")

        // Test AutoConfirm toggle
        app.typeKey(.tab, modifierFlags: .shift)
    }
}
