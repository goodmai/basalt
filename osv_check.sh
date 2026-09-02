#!/bin/bash
# Query OSV for every pinned dependency in Package.resolved.
#
# OSV's SwiftPM ecosystem is "SwiftURL", and it keys packages by their repository
# URL without the scheme — "github.com/apple/swift-nio", not "swift-nio". The
# previous version of this script asked for the bare name, which matches nothing:
# every query came back {} and the scan passed by never checking anything.
#
# Exits 1 when a dependency is affected, so CI stops rather than filing it in a
# report nobody reads.
set -uo pipefail

RESOLVED="${1:-Package.resolved}"
if [[ ! -f "$RESOLVED" ]]; then
    echo "No $RESOLVED — nothing to scan" >&2
    exit 1
fi

found=0
while IFS=$'\t' read -r name url version; do
    [[ -z "$url" || -z "$version" ]] && continue
    pkg="${url#https://}"
    pkg="${pkg%.git}"

    response=$(curl -sS -m 30 "https://api.osv.dev/v1/query" \
        -d "{\"package\":{\"name\":\"$pkg\",\"ecosystem\":\"SwiftURL\"},\"version\":\"$version\"}")

    ids=$(printf '%s' "$response" | jq -r '.vulns[]?.id' 2>/dev/null)
    if [[ -n "$ids" ]]; then
        found=1
        echo "VULNERABLE  $name @ $version"
        printf '%s' "$response" | jq -r '.vulns[] | "    \(.id)  \(.summary // "no summary")"'
    else
        echo "ok          $name @ $version"
    fi
done < <(python3 -c '
import json, sys
pins = json.load(open(sys.argv[1]))["pins"]
for p in pins:
    state = p.get("state", {})
    print("\t".join([p.get("identity", "?"), p.get("location", ""), state.get("version") or state.get("revision", "")]))
' "$RESOLVED")

if (( found )); then
    echo
    echo "OSV reported vulnerable dependencies — see above."
    exit 1
fi

echo
echo "OSV: no known vulnerabilities in the pinned dependencies."
