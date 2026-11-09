import Foundation
import IOKit.ps

public enum PowerSource {
    public static func current() -> PowerState {
        guard let blob = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let sources = IOPSCopyPowerSourcesList(blob)?.takeRetainedValue() as? [CFTypeRef]
        else { return .desktop }
        let battery = sources.lazy.compactMap { IOPSGetPowerSourceDescription(blob, $0)?.takeUnretainedValue() as? [String: Any] }
            .first { ($0[kIOPSTypeKey] as? String) == kIOPSInternalBatteryType }
        let adapter = IOPSCopyExternalPowerAdapterDetails()?.takeRetainedValue() as? [String: Any]
        guard let battery else { return .desktop }
        return state(battery: battery, adapter: adapter)
    }

    static func state(battery: [String: Any], adapter: [String: Any]?) -> PowerState {
        PowerState(
            onAC: (battery[kIOPSPowerSourceStateKey] as? String) == kIOPSACPowerValue,
            charging: battery[kIOPSIsChargingKey] as? Bool ?? false,
            batteryPercent: battery[kIOPSCurrentCapacityKey] as? Int,
            adapterWatts: adapter?[kIOPSPowerAdapterWattsKey] as? Int)
    }
}
