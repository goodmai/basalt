import Foundation
import Testing
@testable import BasaltCore

@Suite("Documentation auto-translation core and pipeline (#720)")
struct DocsTranslationCoreTests {
    @Test("DocLocale maps README and docs paths for all locales including ja, ko, and id")
    func localePathsAndMetadata() {
        #expect(DocLocale.allCases.contains(.ja))
        #expect(DocLocale.allCases.contains(.ko))
        #expect(DocLocale.allCases.contains(.id))
        #expect(DocLocale.ko.nativeName == "한국어")
        #expect(DocLocale.id.nativeName == "Bahasa Indonesia")
        #expect(DocLocale.ja.nativeName == "日本語")
        #expect(DocLocale.ko.targetPath(forEnglishPath: "README.md") == "README.ko.md")
        #expect(DocLocale.id.targetPath(forEnglishPath: "docs/install.md") == "docs/i18n/id/install.md")
        #expect(DocLocale.ja.targetPath(forEnglishPath: "compose.md") == "docs/i18n/ja/compose.md")
        #expect(DocLocale.canonicalDocs.count == 13)
    }

    @Test("MarkdownSegmenter preserves fenced code blocks, tilde fences, and shields badges verbatim")
    func segmenterProtectsCodeAndBadges() {
        let md = """
        <p align="center"><img src="images/logo.svg" width="128"></p>
        # Title
        [![License](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)

        Intro text with `basalt serve --rest`.

        ```bash
        basalt serve --model mlx-community/translategemma-12b-it-4bit --rest
        ```

        ~~~yaml
        version: 2
        ~~~

        Closing paragraph.
        """
        let segments = MarkdownSegmenter.split(md)
        #expect(MarkdownSegmenter.join(segments) == md)
        let protectedTexts = segments.compactMap { seg -> String? in
            if case .protected(let text) = seg { return text }
            return nil
        }
        #expect(protectedTexts.contains(where: { $0.contains("basalt serve --model") }))
        #expect(protectedTexts.contains(where: { $0.contains("version: 2") }))
        #expect(protectedTexts.contains(where: { $0.contains("img.shields.io") }))
        #expect(protectedTexts.contains(where: { $0.contains("<img src=\"images/logo.svg\"") }))
    }

    @Test("MarkdownSegmenter.maskInline protects inline code and URLs and fails on dropped placeholder")
    func inlinePlaceholderRoundTripAndTamperDetection() throws {
        let line = "Run `basalt fit` or visit https://bas-alt.vercel.app/docs for details."
        let protected = MarkdownSegmenter.maskInline(line)
        #expect(protected.tokens.count == 2)
        #expect(!protected.masked.contains("basalt fit"))
        #expect(!protected.masked.contains("https://bas-alt.vercel.app/docs"))

        let translated = protected.masked.replacingOccurrences(of: "Run", with: "실행")
        let restored = try protected.restore(in: translated)
        #expect(restored.contains("`basalt fit`"))
        #expect(restored.contains("https://bas-alt.vercel.app/docs"))
        #expect(restored.contains("실행"))

        let mangled = translated.replacingOccurrences(of: "__BASALT_PH_0__", with: "")
        #expect(throws: DocTranslationError.missingPlaceholder("__BASALT_PH_0__")) {
            _ = try protected.restore(in: mangled)
        }
    }

    @Test("DocTranslationManifest detects missing files and English source digest changes")
    func manifestDetectsStaleAndUpToDateTargets() throws {
        var manifest = DocTranslationManifest()
        let sources = ["README.md": "# Hello v1", "docs/install.md": "# Install v1"]
        let initialStale = manifest.staleTargets(
            englishSources: sources,
            locales: [.ja, .ko, .id],
            existingTargets: ["README.ja.md"]
        )
        #expect(initialStale.count == 6)

        manifest.record(sourcePath: "README.md", sourceText: "# Hello v1", locale: .ja)
        let afterRecord = manifest.staleTargets(
            englishSources: sources,
            locales: [.ja, .ko, .id],
            existingTargets: ["README.ja.md"]
        )
        #expect(afterRecord.count == 5)
        #expect(!afterRecord.contains(where: { $0.source == "README.md" && $0.locale == .ja }))

        let updatedSources = ["README.md": "# Hello v2", "docs/install.md": "# Install v1"]
        let afterEdit = manifest.staleTargets(
            englishSources: updatedSources,
            locales: [.ja],
            existingTargets: ["README.ja.md", "docs/i18n/ja/install.md"]
        )
        #expect(afterEdit.contains(where: { $0.source == "README.md" && $0.locale == .ja }))

        let encoded = try JSONEncoder().encode(manifest)
        let decoded = try JSONDecoder().decode(DocTranslationManifest.self, from: encoded)
        #expect(decoded == manifest)
    }

    @Test("DocLinkRewriter renders language bars with ko and id and rewrites README doc links")
    func linkRewriterAndLanguageBars() {
        let koBar = DocLinkRewriter.readmeLanguageBar(active: .ko)
        #expect(koBar.contains("[English](README.md)"))
        #expect(koBar.contains("**한국어**"))
        #expect(koBar.contains("[Bahasa Indonesia](README.id.md)"))
        #expect(koBar.contains("[日本語](README.ja.md)"))

        let idDocBar = DocLinkRewriter.docLanguageBar(docFilename: "install.md", active: .id)
        #expect(idDocBar.contains("[English](../../install.md)"))
        #expect(idDocBar.contains("**Bahasa Indonesia**"))
        #expect(idDocBar.contains("[한국어](../ko/install.md)"))

        let rewritten = DocLinkRewriter.rewriteReadmeLinks("See [install](docs/install.md#requirements) and [models](docs/models.md).", locale: .ko)
        #expect(rewritten.contains("(docs/i18n/ko/install.md#requirements)"))
        #expect(rewritten.contains("(docs/i18n/ko/models.md)"))
    }

    @Test("DocTranslationPipeline translates prose while keeping code fences, inline code, URLs, and rewritten links intact")
    func endToEndMockTranslationPipeline() async throws {
        let english = """
        # Basalt Guide

        Install with `brew install goodmai/basalt/basalt` and see [install](docs/install.md).

        ```bash
        basalt compose up docs/examples/compose/translate.yaml
        ```
        """
        let prompt = TranslationPromptBuilder.systemPrompt(target: .id)
        #expect(prompt.contains("Indonesian"))
        #expect(prompt.contains("__BASALT_PH_N__"))
        #expect(TranslationPromptBuilder.userPrompt(maskedMarkdown: "Hi", target: .ko).contains("Korean"))

        let translated = try await DocTranslationPipeline.translateDocument(
            english,
            sourcePath: "README.md",
            to: .id
        ) { masked, locale in
            "[\(locale.rawValue)] " + masked
        }
        #expect(translated.contains("[id] # Basalt Guide"))
        #expect(translated.contains("`brew install goodmai/basalt/basalt`"))
        #expect(translated.contains("(docs/i18n/id/install.md)"))
        #expect(translated.contains("basalt compose up docs/examples/compose/translate.yaml"))

        await #expect(throws: DocTranslationError.emptyTranslation) {
            _ = try await DocTranslationPipeline.translateDocument("Hello", sourcePath: "docs/install.md", to: .ko) { _, _ in "   " }
        }
    }
}
