import Foundation
import ArgumentParser
import BasaltCore

extension ServeCommand {
    func harnessRoutingTable() throws -> HarnessRoutingTable {
        let adapterSpecs = try Self.adapterSources(adapters)
        return try Self.parseHarnessRoutingTable(
            harnessAdapters: harnessAdapters,
            defaultAdapter: defaultAdapter,
            registeredAdapters: Set(adapterSpecs.keys)
        )
    }

    func validateHarnessRouting(registeredAdapters: Set<String>) throws {
        _ = try Self.parseHarnessRoutingTable(
            harnessAdapters: harnessAdapters,
            defaultAdapter: defaultAdapter,
            registeredAdapters: registeredAdapters
        )
    }

    static func parseHarnessRoutingTable(
        harnessAdapters: [String],
        defaultAdapter: String?,
        registeredAdapters: Set<String>
    ) throws -> HarnessRoutingTable {
        do throws(BasaltError) {
            let mappings = try HarnessRoutingTable.parseMappings(harnessAdapters)
            let table = HarnessRoutingTable(
                harnessToAdapter: mappings,
                defaultAdapter: defaultAdapter?.trimmingCharacters(in: .whitespacesAndNewlines)
            )
            try table.validate(registeredAdapters: registeredAdapters)
            return table
        } catch {
            throw ValidationError(error.errorDescription ?? error.localizedDescription)
        }
    }
}
