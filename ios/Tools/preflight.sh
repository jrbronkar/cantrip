#!/bin/sh
set -eu

if [ "$#" -ne 2 ] && [ "$#" -ne 3 ]; then
    printf 'Usage: sh ios/Tools/preflight.sh CHECKPOINT_DIRECTORY LAYER_ZERO_FIXTURE [EXPERT_REFERENCE]\n' >&2
    exit 2
fi
checkpoint=$1
fixture=$2
root=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
scratch=$(mktemp -d "${TMPDIR:-/tmp}/inkling-preflight.XXXXXX")
trap 'rm -rf "$scratch"' EXIT
export HF_HUB_OFFLINE=1

"$root/.venv/bin/python" -B "$root/ios/Tests/context_reference.py" "$checkpoint" "$scratch/context.json"
xcrun swiftc -O -module-cache-path "$scratch/swift-cache" \
    "$root/ios/Sources/EmbeddingProbe.swift" "$root/ios/Sources/ConversationContext.swift" \
    "$root/ios/Tests/Preflight.swift" -o "$scratch/preflight"
if [ "$#" -eq 3 ]; then
    "$scratch/preflight" "$fixture" "$scratch/context.json" "$checkpoint/layer_00_packed.npz" "$3"
else
    "$scratch/preflight" "$fixture" "$scratch/context.json"
fi

if ! xcodebuild -project "$root/ios/InklingProbe.xcodeproj" -scheme InklingProbe \
    -configuration Release -destination 'generic/platform=iOS' \
    -derivedDataPath "$scratch/derived" CODE_SIGNING_ALLOWED=NO ALWAYS_SEARCH_USER_PATHS=NO \
    build > "$scratch/build.log" 2>&1; then
    tail -80 "$scratch/build.log" >&2
    exit 1
fi
app="$scratch/derived/Build/Products/Release-iphoneos/InklingProbe.app"
test -f "$app/LookAtMyQuant-MIT.txt"
test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$app/Info.plist")" = 0.1.0
test "$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$app/Info.plist")" = 1
printf 'Passed unsigned iPhone Release build and bundle checks\n'
