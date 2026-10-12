#!/usr/bin/env zsh
# Publish the Release build as the NAMTRIX website's Player download.
#
#   ./script/publish_player.sh
#
# The website's "Download NAMTRIX Player" button (namtrix.ummardesom.com and
# the NAMTRIX apps) always points at
#   https://namtrix.ummardesom.com/download/NAMTRIX-Player-macOS.zip
# which is served from the R2 bucket "namtrix", key "NAMTRIX-Player-macOS.zip".
# Replacing that one object publishes a new Player; nothing on the website
# changes. Needs wrangler logged in (npx wrangler whoami).
#
# Build the Release first. This script refuses to publish a build older than
# the newest source change under plugin/.
set -euo pipefail
cd "$(dirname "$0")/.."

release_dir="build/plugin/TONE3000_artefacts/Release"
bucket_key="namtrix/NAMTRIX-Player-macOS.zip"
live_url="https://namtrix.ummardesom.com/download/NAMTRIX-Player-macOS.zip"
zip_name="NAMTRIX-Player-macOS.zip"
folder_name="NAMTRIX Player"

bundles=(
  "VST3/NAMTRIX Player.vst3"
  "AU/NAMTRIX Player.component"
  "Standalone/NAMTRIX Player.app"
)

die() {
  echo "publish_player: $*" >&2
  exit 1
}

# The version people see in the Player's header (core/Namtrix.h), not the
# VERSION file: that still carries upstream TONE3000's number, which drives
# their installer and update check.
version="$(sed -nE 's/.*kVersion = "([^"]+)".*/\1/p' plugin/ui/core/Namtrix.h)"
[ -n "$version" ] || die "couldn't read the Player version from plugin/ui/core/Namtrix.h"

# 1. All three builds present, and none older than the source they came from.
#
# Freshness is judged on each bundle's main executable: the bundle directory's
# own timestamp doesn't move when the linker rewrites the binary inside it.
# Sources are what git considers part of plugin/ -- tracked files, submodule
# files and untracked-but-not-ignored ones -- so build output under plugin/
# (the UI testbed's build folder) can't make a fresh build look stale.
# A tracked file can be missing on disk -- the build deletes a nested
# submodule's build/.gitignore -- and stat failing on it makes xargs exit
# non-zero, which pipefail would turn into a silent abort. A file that isn't
# there can't be newer than the build, so its failure is ignored.
newest_src="$(
  { git ls-files -z --recurse-submodules plugin
    git ls-files -z --others --exclude-standard plugin
  } | { xargs -0 stat -f '%m %N' 2>/dev/null || true; } | sort -n | tail -1
)"
[ -n "$newest_src" ] || die "couldn't list the sources under plugin/"
newest_src_time="${newest_src%% *}"
newest_src_file="${newest_src#* }"

# (Not `path`: in zsh that's the array tied to $PATH, and assigning it would
# leave no commands to run.)
for bundle in "${bundles[@]}"; do
  bundle_path="$release_dir/$bundle"
  [ -d "$bundle_path" ] || die "missing build: $bundle_path
Build the Release first (see README, Quick start)."
  binary="$bundle_path/Contents/MacOS/NAMTRIX Player"
  [ -f "$binary" ] || die "build has no executable: $binary"
  if [ "$(stat -f %m "$binary")" -lt "$newest_src_time" ]; then
    die "stale build: $bundle
  built   $(stat -f '%Sm' "$binary")
  source  $(stat -f '%Sm' "$newest_src_file")  ($newest_src_file)
Rebuild the Release before publishing."
  fi
  # The READ ME and the summary name this version, so the build has to carry
  # it. Process substitution, not `strings | grep -q`: grep quitting early
  # would SIGPIPE strings, and pipefail would read a match as a failure.
  grep -qxF -- "$version" <(strings "$binary") ||
    die "$bundle doesn't contain $version (from plugin/ui/core/Namtrix.h). Rebuild the Release."
done

work="$(mktemp -d -t namtrix-publish)"
trap 'rm -rf "$work"' EXIT
stage="$work/$folder_name"
mkdir -p "$stage"

# 2-3. Copy, then re-sign each copy ad hoc.
#
# The AU and standalone come out of the build with a signature that claims
# resources it doesn't seal ("code has no resources but signature indicates
# they must be present"). That runs on the build machine, but a downloaded
# copy fails to load on anyone else's Mac. A fresh deep ad-hoc signature fixes
# it; the strict check proves it before anything is zipped.
for bundle in "${bundles[@]}"; do
  copy="$stage/${bundle:t}"
  ditto "$release_dir/$bundle" "$copy"
  codesign --force --deep --sign - "$copy"
done
for bundle in "${bundles[@]}"; do
  copy="$stage/${bundle:t}"
  codesign --verify --deep --strict "$copy" ||
    die "signature check failed after re-signing: ${bundle:t}"
done

# 4. READ ME FIRST.txt
cat > "$stage/READ ME FIRST.txt" <<EOF
NAMTRIX Player $version
==============================

A plugin and standalone app that plays Neural Amp Modeler (.nam) models,
including parametric ones: models captured across an amp's controls, whose
knobs work in the Player the way they do on the real amp.

Formats
  VST3, AU, Standalone

Requirements
  Apple silicon Mac (M1 or later), macOS 11 or later.

Install
  NAMTRIX Player.vst3       ->  /Library/Audio/Plug-Ins/VST3/
  NAMTRIX Player.component  ->  /Library/Audio/Plug-Ins/Components/
  NAMTRIX Player.app        ->  /Applications/

The Player isn't notarized by Apple, so macOS blocks it until you clear the
quarantine flag. After copying the files above, run this once in Terminal
(it asks for your password):

  sudo xattr -dr com.apple.quarantine "/Library/Audio/Plug-Ins/VST3/NAMTRIX Player.vst3" "/Library/Audio/Plug-Ins/Components/NAMTRIX Player.component" "/Applications/NAMTRIX Player.app"

Then restart your DAW.

Credits
  Based on the TONE3000 plugin (MIT license) and NeuralAmpModelerCoreParametric.
EOF

# 5. Zip, then prove the zip itself: unpack a copy and re-run the strict check
# on what people will actually download, not on the staging folder.
zip="$work/$zip_name"
(cd "$work" && ditto -c -k --sequesterRsrc --keepParent "$folder_name" "$zip_name")
unpacked="$work/verify"
mkdir -p "$unpacked"
ditto -x -k "$zip" "$unpacked"
for bundle in "${bundles[@]}"; do
  codesign --verify --deep --strict "$unpacked/$folder_name/${bundle:t}" ||
    die "signature check failed inside the zip: ${bundle:t}"
done
size="$(stat -f %z "$zip")"

# 6. Upload: replacing this one object is the whole publish.
npx --yes wrangler r2 object put "$bucket_key" --file "$zip" \
  --content-type application/zip --remote

# 7. Confirm the live download is the zip just uploaded. A few retries, in case
# the edge is briefly still serving the previous object.
live_status=""
live_length=""
for attempt in 1 2 3 4 5 6; do
  headers="$(curl -sI "$live_url" | tr -d '\r')"
  live_status="$(echo "$headers" | awk 'NR==1 {print $2}')"
  live_length="$(echo "$headers" | awk 'tolower($1)=="content-length:" {print $2}')"
  [ "$live_status" = "200" ] && [ "$live_length" = "$size" ] && break
  sleep 5
done
[ "$live_status" = "200" ] || die "live URL returned ${live_status:-no response}: $live_url"
[ "$live_length" = "$size" ] || die "live URL serves $live_length bytes, uploaded $size: $live_url
The upload succeeded; the edge may still be caching the previous file."

echo
echo "Published NAMTRIX Player $version"
echo "  size  $size bytes ($(( size / 1024 / 1024 )) MB)"
echo "  url   $live_url"
