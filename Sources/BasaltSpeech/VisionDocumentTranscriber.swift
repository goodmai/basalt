import Foundation
import Vision
import ImageIO
import BasaltCore

public struct VisionDocumentTranscriber: DocumentTranscriber {
    public init() {}

    public func transcribe(_ image: ImageInput) async throws(BasaltError) -> DocumentMarkdown {
        guard let source = CGImageSourceCreateWithData(image.data as CFData, nil),
              let page = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw .invalidRequestStructure(details: "the page image cannot be decoded")
        }
        let regions = (try? await RecognizeDocumentsRequest().perform(on: page).flatMap { Self.regions(of: $0.document) }) ?? []
        guard regions.isEmpty else { return DocumentMarkdown(regions) }
        do {
            return DocumentMarkdown(try await Self.lineRegions(in: page))
        } catch {
            throw .inferenceError("Vision could not read the image: \(error.localizedDescription)")
        }
    }

    static func regions(of document: DocumentObservation.Container) -> [DocumentRegion] {
        let title = document.title.map { DocumentRegion(.title($0.transcript), box: box($0.boundingRegion.boundingBox)) }
        let paragraphs = document.paragraphs.map { DocumentRegion(.paragraph($0.transcript), box: box($0.boundingRegion.boundingBox)) }
        let lists = document.lists.map { list in
            DocumentRegion(
                .list(ordered: list.items.contains { isOrdered($0.markerType) }, items: list.items.map(\.itemString)),
                box: box(list.boundingRegion.boundingBox))
        }
        let tables = document.tables.map { table in
            let cells = table.rows.joined().map { cell in
                DocumentRegion.Cell(row: cell.rowRange.lowerBound, column: cell.columnRange.lowerBound, text: cell.content.text.transcript)
            }
            return DocumentRegion(.table(cells), box: box(table.boundingRegion.boundingBox))
        }
        return (title.map { [$0] } ?? []) + paragraphs + lists + tables
    }

    static func lineRegions(in page: CGImage) async throws -> [DocumentRegion] {
        try await RecognizeTextRequest().perform(on: page).compactMap { line in
            line.topCandidates(1).first.map { DocumentRegion(.paragraph($0.string), box: box(line.boundingBox)) }
        }
    }

    static func isOrdered(_ marker: DocumentObservation.Container.List.Marker?) -> Bool {
        switch marker {
        case .decimal?, .decorativeDecimal?, .compositeDecimal?, .lowercaseLatin?, .uppercaseLatin?: true
        default: false
        }
    }

    static func box(_ rect: NormalizedRect) -> OCRBox {
        let frame = rect.cgRect
        return OCRBox(x1: frame.minX, y1: 1 - frame.maxY, x2: frame.maxX, y2: 1 - frame.minY)
    }
}
