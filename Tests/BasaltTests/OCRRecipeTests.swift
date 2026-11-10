import Testing
import Foundation
import CoreGraphics
import CoreText
import ImageIO
import PDFKit
@testable import BasaltCore
@testable import BasaltInference

enum OCRFixture {
    static func pdf(pages: [String]) -> Data {
        let data = NSMutableData()
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        guard let consumer = CGDataConsumer(data: data), let context = CGContext(consumer: consumer, mediaBox: &box, nil) else { return Data() }
        for text in pages {
            context.beginPDFPage(nil)
            context.setFillColor(CGColor(red: 0, green: 0, blue: 0, alpha: 1))
            context.fill(CGRect(x: 36, y: 792 - 136, width: 100, height: 100))
            let line = CTLineCreateWithAttributedString(NSAttributedString(
                string: text, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName("Helvetica" as CFString, 24, nil)]))
            context.textPosition = CGPoint(x: 72, y: 500)
            CTLineDraw(line, context)
            context.endPDFPage()
        }
        context.closePDF()
        return data as Data
    }

    static func image(_ rows: [[(UInt8, UInt8, UInt8)]]) -> CGImage? {
        let bytes: [UInt8] = rows.flatMap { row in row.flatMap { pixel -> [UInt8] in [pixel.0, pixel.1, pixel.2, 255] } }
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        return CGImage(
            width: rows[0].count, height: rows.count, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: rows[0].count * 4,
            space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
            provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }

    static func png(width: Int, height: Int) throws -> Data {
        let image = try #require(image(Array(repeating: Array(repeating: (255, 255, 255), count: width), count: height)))
        return try OCRImage.png(image)
    }

    static func pixel(_ image: CGImage, x: Int, y: Int) -> (r: UInt8, g: UInt8, b: UInt8)? {
        guard let copy = OCRImage.resized(image, width: image.width, height: image.height),
              let data = copy.dataProvider?.data as Data? else { return nil }
        let offset = y * copy.bytesPerRow + x * 4
        return (data[offset], data[offset + 1], data[offset + 2])
    }

    static func size(of input: ImageInput) throws -> (Int, Int) {
        let image = try OCRImage.decode(input.data)
        return (image.width, image.height)
    }
}

@Suite("OCR recipes: families, prompts and requests (#400)")
struct OCRRecipeTests {
    @Test("glm_ocr is GLM-OCR; qwen2_vl is MinerU2.5 only with a mineru repo hint; other families have no recipe")
    func detectsFamilies() {
        #expect(OCRRecipe.of(modelType: "glm_ocr", hints: [])?.family == .glmOCR)
        #expect(OCRRecipe.of(modelType: "qwen2_vl", hints: ["/hub/models--opendatalab--MinerU2.5-Pro-2605-1.2B/snapshots/abc"])?.family == .minerU)
        #expect(OCRRecipe.of(modelType: "qwen2_vl", hints: ["/hub/models--Qwen--Qwen2-VL-2B-Instruct", ""]) == nil)
        #expect(OCRRecipe.of(modelType: "llama", hints: ["mineru"]) == nil)
        #expect(OCRRecipe.of(modelType: nil, hints: ["GLM-OCR"]) == nil)
    }

    @Test("GLM-OCR uses its three published prompts; document is page-level text recognition; chart is refused",
          arguments: [(OCRTask.document, "Text Recognition:", "text"), (.text, "Text Recognition:", "text"),
                      (.table, "Table Recognition:", "table"), (.formula, "Formula Recognition:", "equation")])
    func glmPrompts(_ task: OCRTask, _ prompt: String, _ type: String) throws {
        #expect(try OCRRecipe(family: .glmOCR).step(for: task) == .recognize(prompt: prompt, blockType: type))
        #expect(throws: BasaltError.self) { try OCRRecipe(family: .glmOCR).step(for: .chart) }
    }

    @Test("MinerU2.5 runs layout detection at 1036x1036 for a document and the mineru-vl-utils prompts for single tasks")
    func minerUSteps() throws {
        let recipe = OCRRecipe(family: .minerU)
        #expect(try recipe.step(for: .document) == .layout(prompt: "\nLayout Detection:", side: 1036))
        #expect(try recipe.step(for: .text) == .recognize(prompt: "\nText Recognition:", blockType: "text"))
        #expect(try recipe.step(for: .table) == .recognize(prompt: "\nTable Recognition:", blockType: "table"))
        #expect(try recipe.step(for: .formula) == .recognize(prompt: "\nFormula Recognition:", blockType: "equation"))
        #expect(try recipe.step(for: .chart) == .recognize(prompt: "\nImage Analysis:", blockType: "chart"))
    }

    @Test("paddleocr_vl is PaddleOCR-VL with its four published element prompts; a document is page-level OCR and its requests are plain greedy",
          arguments: [(OCRTask.document, "OCR:", "text"), (.text, "OCR:", "text"), (.table, "Table Recognition:", "table"),
                      (.formula, "Formula Recognition:", "equation"), (.chart, "Chart Recognition:", "chart")])
    func paddlePrompts(_ task: OCRTask, _ prompt: String, _ type: String) throws {
        let recipe = try #require(OCRRecipe.of(modelType: "paddleocr_vl", hints: []))
        #expect(recipe.family == .paddleOCR)
        #expect(try recipe.step(for: task) == .recognize(prompt: prompt, blockType: type))
        let image = try ImageInput(data: try OCRFixture.png(width: 4, height: 4), mime: "image/png")
        let request = recipe.request(prompt: prompt, image: image, stage: .recognition)
        #expect(request.temperature == 0 && request.repetitionPenalty == nil && request.presencePenalty == nil)
    }

    @Test("deepseekocr is DeepSeek-OCR with its published prompts: a document converts to markdown with grounding, text is Free OCR, a chart is Parse the figure; table and formula are refused",
          arguments: [(OCRTask.document, "<|grounding|>Convert the document to markdown.", "text"), (.text, "Free OCR.", "text"), (.chart, "Parse the figure.", "chart")])
    func deepSeekPrompts(_ task: OCRTask, _ prompt: String, _ type: String) throws {
        let recipe = try #require(OCRRecipe.of(modelType: "deepseekocr", hints: []))
        #expect(recipe.family == .deepSeekOCR && recipe.resolution == nil)
        #expect(try recipe.step(for: task) == .recognize(prompt: prompt, blockType: type))
        #expect(throws: BasaltError.self) { try recipe.step(for: .table) }
        #expect(throws: BasaltError.self) { try recipe.step(for: .formula) }
    }

    @Test("A DeepSeek-OCR request moves the grounding token out of the user text into its OCR mode, with the chosen resolution or Gundam; other families carry no mode and refuse a resolution")
    func deepSeekRequests() throws {
        let image = try ImageInput(data: try OCRFixture.png(width: 4, height: 4), mime: "image/png")
        let grounded = try OCRRecipe(family: .deepSeekOCR).with(resolution: .large)
            .request(prompt: "<|grounding|>Convert the document to markdown.", image: image, stage: .recognition)
        #expect(grounded.messages?.first?.text == "Convert the document to markdown." && grounded.prompt == "Convert the document to markdown.")
        #expect(grounded.ocrMode == OCRMode(resolution: .large, grounding: true))
        #expect(grounded.temperature == 0 && grounded.repetitionPenalty == nil)
        let free = OCRRecipe(family: .deepSeekOCR).request(prompt: "Free OCR.", image: image, stage: .recognition)
        #expect(free.ocrMode == OCRMode(resolution: .gundam, grounding: false) && free.messages?.first?.text == "Free OCR.")
        #expect(OCRRecipe(family: .paddleOCR).request(prompt: "<|grounding|>OCR:", image: image, stage: .recognition).ocrMode == nil)
        #expect(throws: BasaltError.self) { try OCRRecipe(family: .paddleOCR).with(resolution: .base) }
        #expect(try OCRRecipe(family: .glmOCR).with(resolution: nil).family == .glmOCR)
    }

    @Test("DeepSeek-OCR grounding output becomes blocks with boxes scaled from 0-999 to the page and Markdown without the tags; image blocks stay empty; ungrounded output is one page block")
    func deepSeekGrounding() {
        let output = """
            <|ref|>title<|/ref|><|det|>[[99, 0, 999, 99]]<|/det|>
            # Contributing

            <|ref|>image<|/ref|><|det|>[[0, 100, 499, 300], [500, 150, 999, 400]]<|/det|>

            <|ref|>table<|/ref|><|det|>[[10, 500, 2000, 600]]<|/det|>
            <table><tr><td>a</td></tr></table>
            """
        let read = OCRRecipe(family: .deepSeekOCR).read(output, blockType: "text")
        #expect(read.markdown == "# Contributing\n\n<table><tr><td>a</td></tr></table>")
        #expect(read.blocks.map(\.type) == ["title", "image", "table"])
        #expect(read.blocks[0].bbox == OCRBox(x1: 99.0 / 999, y1: 0, x2: 1, y2: 99.0 / 999))
        #expect(read.blocks[1].bbox == OCRBox(x1: 0, y1: 100.0 / 999, x2: 1, y2: 400.0 / 999) && read.blocks[1].text.isEmpty)
        #expect(read.blocks[2].bbox.x2 == 1)
        let plain = OCRRecipe(family: .deepSeekOCR).read(" plain text \n", blockType: "text")
        #expect(plain.markdown == "plain text" && plain.blocks == [OCRBlock(type: "text", bbox: .page, text: "plain text")])
        #expect(OCRRecipe(family: .deepSeekOCR).read("a chart", blockType: "chart").blocks.map(\.type) == ["chart"])
        #expect(DeepSeekGrounding.box("[[1, 2, 3]]") == .page)
        #expect(OCRRecipe(family: .paddleOCR).read(" x ", blockType: "text").blocks == [OCRBlock(type: "text", bbox: .page, text: "x")])
    }

    @Test("A request's resolution parses to a mode and an unknown one is refused at the boundary")
    func resolutionParsing() throws {
        let document = OCRDocument.image(try ImageInput(data: try OCRFixture.png(width: 4, height: 4), mime: "image/png"))
        #expect(try OCRRequest(document: document, pages: nil, task: nil, format: nil, model: nil, resolution: "tiny").resolution == .tiny)
        #expect(try OCRRequest(document: document, pages: nil, task: nil, format: nil, model: nil).resolution == nil)
        #expect(throws: BasaltError.self) { try OCRRequest(document: document, pages: nil, task: nil, format: nil, model: nil, resolution: "huge") }
    }

    @Test("A recipe request is greedy, carries the page as the only image, and takes the family's repetition settings")
    func requests() throws {
        let image = try ImageInput(data: try OCRFixture.png(width: 4, height: 4), mime: "image/png")
        let glm = OCRRecipe(family: .glmOCR).request(prompt: "Text Recognition:", image: image, stage: .recognition)
        #expect(glm.temperature == 0 && glm.repetitionPenalty == 1.1 && glm.presencePenalty == nil && glm.maxTokens == 8192)
        #expect(glm.messages?.count == 1 && glm.messages?.first?.text == "Text Recognition:" && glm.messages?.first?.images == [image])
        let layout = OCRRecipe(family: .minerU).request(prompt: "\nLayout Detection:", image: image, stage: .layout)
        #expect(layout.presencePenalty == nil && layout.repetitionPenalty == nil && layout.temperature == 0)
        #expect(OCRRecipe(family: .minerU).request(prompt: "\nText Recognition:", image: image, stage: .recognition).presencePenalty == 1)
    }

    @Test("Post-processing: MinerU and PaddleOCR-VL tables turn from OTSL into HTML, MinerU equations get display brackets; GLM-OCR output is only trimmed")
    func postProcessing() {
        let minerU = OCRRecipe(family: .minerU)
        #expect(minerU.finished(" <fcel>a<fcel>b<nl> ", blockType: "table") == "<table><tr><td>a</td><td>b</td></tr></table>")
        #expect(minerU.finished("x^2", blockType: "equation") == "\\[\nx^2\n\\]")
        #expect(minerU.finished("\\[x\\]", blockType: "equation") == "\\[x\\]")
        #expect(minerU.finished("", blockType: "equation") == "")
        #expect(OCRRecipe(family: .glmOCR).finished("\n<fcel>a\n", blockType: "table") == "<fcel>a")
        let paddle = OCRRecipe(family: .paddleOCR)
        #expect(paddle.finished("<fcel>Model<fcel>Size<nl><fcel>GLM-OCR<fcel>0.9B<nl>", blockType: "table")
            == "<table><tr><td>Model</td><td>Size</td></tr><tr><td>GLM-OCR</td><td>0.9B</td></tr></table>")
        #expect(paddle.finished(" x^2 ", blockType: "equation") == "x^2")
    }

    @Test("glm_ocr and qwen2_vl stay unverified families; paddleocr_vl verifies lookup, batched decode, prompt-cache reuse and grammar jump-forward on text turns; no MTP, CacheBlend, quantized KV, KV zones, head split or Quest")
    func capabilities() {
        #expect(ModelCapabilities.of(modelType: "glm_ocr") == nil)
        #expect(ModelCapabilities.of(modelType: "qwen2_vl") == nil)
        #expect(ModelCapabilities.of(modelType: "paddleocr_vl")?.verified == Set(ModelCapabilities.Feature.allCases).subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))
        #expect(ModelCapabilities.of(modelType: "deepseekocr")?.verified == Set(ModelCapabilities.Feature.allCases).subtracting([.mtp, .cacheBlend, .kvQuantization, .kvZones, .kvHeadSplit, .quest]))
    }
}

@Suite("OCR pages: page ranges, documents and PDF rendering (#400)")
struct OCRPageTests {
    @Test("Page ranges parse to sorted unique page numbers", arguments: [("1-3,5", [1, 2, 3, 5]), (" 2 , 2-3", [2, 3]), ("7", [7])])
    func parsesRanges(_ text: String, _ numbers: [Int]) throws {
        #expect(try OCRPageSelection(text).numbers == numbers)
    }

    @Test("Malformed, reversed, zero and oversized ranges are refused without expanding them",
          arguments: ["", "0", "3-1", "a", "1-", "-2", "1-2-3", "1,,2", "1-33", "1-1000000000", "1-20,21-40"])
    func refusesRanges(_ text: String) {
        #expect(throws: BasaltError.self) { try OCRPageSelection(text) }
    }

    @Test("Without pages a document reads every page up to the limit; a page past the end is refused")
    func resolvesPages() throws {
        #expect(try OCRPageSelection.resolve(nil, pageCount: 3) == [1, 2, 3])
        #expect(throws: BasaltError.self) { try OCRPageSelection.resolve(nil, pageCount: 40) }
        #expect(throws: BasaltError.self) { try OCRPageSelection.resolve(OCRPageSelection("2-4"), pageCount: 3) }
    }

    @Test("A document is sniffed from its bytes: PDF, an image with its MIME type, or refused; base64 and data: URIs decode")
    func sniffsDocuments() throws {
        let png = try OCRFixture.png(width: 2, height: 2)
        guard case .pdf = try OCRDocument(data: OCRFixture.pdf(pages: ["a"])) else { Issue.record("not a PDF"); return }
        guard case .image(let image) = try OCRDocument(encoded: "data:image/png;base64," + png.base64EncodedString()) else {
            Issue.record("not an image")
            return
        }
        #expect(image.mime == "image/png")
        #expect(throws: BasaltError.self) { try OCRDocument(data: Data("hello".utf8)) }
        #expect(throws: BasaltError.self) { try OCRDocument(data: Data()) }
        #expect(throws: BasaltError.self) { try OCRDocument(encoded: "%%%") }
        #expect(throws: BasaltError.self) { try OCRDocument(data: png).pages(OCRPageSelection("2"), dpi: 200) }
    }

    @Test("A generated PDF renders the selected pages to PNG at the target DPI, upright, with its text layer intact")
    func rendersPDF() throws {
        let pdf = OCRFixture.pdf(pages: ["Hello OCR", "Second page"])
        #expect(PDFDocument(data: pdf)?.page(at: 1)?.string?.contains("Second page") == true)
        let pages = try OCRImage.render(pdf: pdf, pages: OCRPageSelection("2"), dpi: 200)
        #expect(pages.map(\.number) == [2])
        #expect(pages.first?.image.mime == "image/png")
        let image = try OCRImage.decode(try #require(pages.first).image.data)
        #expect((image.width, image.height) == (1700, 2200))
        let mark = try #require(OCRFixture.pixel(image, x: 86 * 200 / 72, y: 86 * 200 / 72))
        #expect(mark.r < 50 && mark.g < 50 && mark.b < 50)
        let blank = try #require(OCRFixture.pixel(image, x: 1600, y: 2100))
        #expect(blank.r > 200 && blank.g > 200 && blank.b > 200)
        #expect(try OCRImage.render(pdf: pdf, pages: nil, dpi: 72).map(\.number) == [1, 2])
        #expect(throws: BasaltError.self) { try OCRImage.render(pdf: pdf, pages: OCRPageSelection("3"), dpi: 72) }
        #expect(throws: BasaltError.self) { try OCRImage.render(pdf: Data("%PDF-1.4 broken".utf8), pages: nil, dpi: 72) }
    }

    @Test("A rotated page renders with its sides swapped, and the long side is capped at 4096 pixels")
    func rendersRotatedAndCapped() throws {
        let page = try #require(PDFDocument(data: OCRFixture.pdf(pages: ["x"]))?.page(at: 0))
        page.rotation = 90
        let rotated = try #require(OCRImage.render(page, dpi: 72))
        #expect((rotated.width, rotated.height) == (792, 612))
        let capped = try #require(OCRImage.render(page, dpi: 1200))
        #expect(max(capped.width, capped.height) == 4096)
    }
}

@Suite("OCR images: crops, rotation and MinerU sizing (#400)")
struct OCRImageTests {
    static let red: (UInt8, UInt8, UInt8) = (255, 0, 0)
    static let blue: (UInt8, UInt8, UInt8) = (0, 0, 255)
    static let green: (UInt8, UInt8, UInt8) = (0, 255, 0)
    static let white: (UInt8, UInt8, UInt8) = (255, 255, 255)

    func color(_ image: CGImage?, _ x: Int, _ y: Int) -> String {
        guard let image, let pixel = OCRFixture.pixel(image, x: x, y: y) else { return "none" }
        return pixel.r > 200 ? (pixel.g > 200 ? "white" : "red") : pixel.g > 200 ? "green" : pixel.b > 200 ? "blue" : "dark"
    }

    @Test("A normalized box crops from the top-left corner of the page")
    func cropsFromTopLeft() throws {
        let image = try #require(OCRFixture.image([[Self.red, Self.blue], [Self.green, Self.white]]))
        #expect(color(OCRImage.crop(image, to: OCRBox(x1: 0, y1: 0, x2: 0.5, y2: 0.5)), 0, 0) == "red")
        #expect(color(OCRImage.crop(image, to: OCRBox(x1: 0.5, y1: 0, x2: 1, y2: 0.5)), 0, 0) == "blue")
        #expect(color(OCRImage.crop(image, to: OCRBox(x1: 0, y1: 0.5, x2: 0.5, y2: 1)), 0, 0) == "green")
        #expect(OCRImage.crop(image, to: OCRBox(x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2)) == nil)
    }

    @Test("Rotation turns counterclockwise like PIL's rotate(angle, expand=True)")
    func rotatesCounterclockwise() throws {
        let image = try #require(OCRFixture.image([[Self.red, Self.blue]]))
        let quarter = OCRImage.rotated(image, counterclockwise: 90)
        #expect((quarter?.width, quarter?.height) == (1, 2))
        #expect(color(quarter, 0, 0) == "blue" && color(quarter, 0, 1) == "red")
        let half = OCRImage.rotated(image, counterclockwise: 180)
        #expect(color(half, 0, 0) == "blue" && color(half, 1, 0) == "red")
        let threeQuarters = OCRImage.rotated(image, counterclockwise: 270)
        #expect(color(threeQuarters, 0, 0) == "red" && color(threeQuarters, 0, 1) == "blue")
    }

    @Test("A crop thinner than 1:50 is padded with white, then any side under 28 pixels is scaled up")
    func fitsCrops() throws {
        let strip = try #require(OCRFixture.image([Array(repeating: Self.red, count: 100)]))
        let fitted = try #require(OCRImage.fitted(strip, minEdge: 28, maxRatio: 50))
        #expect((fitted.width, fitted.height) == (1400, 28))
        let small = try #require(OCRFixture.image(Array(repeating: Array(repeating: Self.red, count: 10), count: 10)))
        let grown = try #require(OCRImage.fitted(small, minEdge: 28, maxRatio: 50))
        #expect((grown.width, grown.height) == (28, 28))
        let large = try #require(OCRFixture.image(Array(repeating: Array(repeating: Self.red, count: 30), count: 30)))
        #expect(OCRImage.fitted(large, minEdge: 28, maxRatio: 50)?.width == 30)
    }
}
