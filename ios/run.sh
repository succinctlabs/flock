#!/bin/zsh
# Build the Rust static library and the probe app, then install and launch it.
#
#   ios/run.sh sim  [<simulator udid>]   [-- <app args>]
#   ios/run.sh phone <device udid>        [-- <app args>]
#
# App args: -global N -all_core M -leaf 256,65536 -tower 4,256 (see ContentView).
# Phone runs need a signing team: sign in once in Xcode, or export
# TEAM=<10-char team id>. Find the phone's udid with `xcrun devicectl list devices`.
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-sim}; shift || true
udid=""
if [[ ${1:-} != "--" && -n ${1:-} ]]; then udid=$1; shift; fi
[[ ${1:-} == "--" ]] && shift
app_args=("$@")
bundle=xyz.succinct.flock.FlockProbe
# Match the app's deployment target so the linker does not warn about the
# Rust objects being built for a newer iOS than the app links against.
export IPHONEOS_DEPLOYMENT_TARGET=17.0

case $mode in
  sim)
    cargo build --release -p flock-mobile --target aarch64-apple-ios-sim
    (cd ios/FlockProbe && xcodegen generate --quiet)
    udid=${udid:-$(xcrun simctl list devices available | grep 'iPhone 17 Pro (' | head -1 | sed -E 's/.*\(([-0-9A-F]+)\).*/\1/')}
    xcodebuild -project ios/FlockProbe/FlockProbe.xcodeproj -scheme FlockProbe -configuration Release \
      -destination "platform=iOS Simulator,id=$udid" -derivedDataPath target/xcode -quiet build
    xcrun simctl boot "$udid" 2>/dev/null || true
    xcrun simctl bootstatus "$udid" -b >/dev/null
    xcrun simctl install "$udid" target/xcode/Build/Products/Release-iphonesimulator/FlockProbe.app
    xcrun simctl launch --console --terminate-running-process "$udid" $bundle -exit 1 "${app_args[@]}" 2>&1 | grep --line-buffered '^\[flock\]'
    ;;
  phone)
    [[ -n $udid ]] || { echo "phone mode needs the device udid (xcrun devicectl list devices)"; exit 2; }
    cargo build --release -p flock-mobile --target aarch64-apple-ios
    (cd ios/FlockProbe && xcodegen generate --quiet)
    xcodebuild -project ios/FlockProbe/FlockProbe.xcodeproj -scheme FlockProbe -configuration Release \
      -destination "platform=iOS,id=$udid" -derivedDataPath target/xcode -allowProvisioningUpdates \
      ${TEAM:+DEVELOPMENT_TEAM=$TEAM} -quiet build
    xcrun devicectl device install app --device "$udid" target/xcode/Build/Products/Release-iphoneos/FlockProbe.app
    xcrun devicectl device process launch --device "$udid" --console --terminate-existing $bundle -exit 1 "${app_args[@]}" 2>&1 | grep --line-buffered '^\[flock\]'
    ;;
  *) echo "usage: ios/run.sh sim|phone [udid] [-- app args]"; exit 2;;
esac
