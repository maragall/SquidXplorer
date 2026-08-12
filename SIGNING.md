# Signing and notarising the macOS build

**Short version: the app is already signed, and that is not enough.** Everything in this document
except the final section is done and needs nothing from you. The final section needs your Apple
Developer account and cannot be done by anyone without it.

## What is true today, measured

Built on an M-series Mac from `scripts/hcs-viewer.spec` with no credentials set:

```
$ codesign -dv dist/hcs-viewer.app
Identifier=com.cephla.squidxplorer.hcsviewer
Format=app bundle with Mach-O thin (arm64)
CodeDirectory v=20400 flags=0x2(adhoc) hashes=2038+3
Signature=adhoc
TeamIdentifier=not set

$ codesign --verify --deep --strict dist/hcs-viewer.app
$ echo $?
0

$ spctl -a -vv dist/hcs-viewer.app
dist/hcs-viewer.app: rejected
```

Three things follow, and the middle one is the one people get wrong:

1. **The bundle is ad-hoc signed and internally consistent.** `codesign --verify --deep --strict`
   exits 0. Nothing is broken or missing.
2. **PyInstaller did that by itself, with `codesign_identity=None`.** macOS will not execute an
   unsigned arm64 Mach-O at all, so ad-hoc signing is not optional and was never the gap. "Add
   ad-hoc signing" is therefore not an available improvement: it is already there.
3. **Gatekeeper rejects it anyway.** `spctl` says `rejected`, and it will say that for any ad-hoc
   bundle no matter how it was produced.

So a recipient who downloads this today gets:

> "hcs-viewer" cannot be opened because the developer cannot be verified.

That message comes from the `com.apple.quarantine` attribute the browser attaches to the download,
checked against a signature Gatekeeper cannot trace to a known developer. **Ad-hoc signing does not
remove it.** Only a Developer ID signature plus notarisation does.

## The one workaround, and its cost

A recipient can right-click the app, choose **Open**, and confirm once. Or:

```
xattr -d com.apple.quarantine /Applications/hcs-viewer.app
```

This is fine for you and for a demo on your own machine. **Do not put it in a customer email.**
Teaching MD Anderson to strip quarantine attributes trains them to bypass the exact control that
protects them from a tampered download, and it will not survive their IT review.

## What is wired up and waiting

`scripts/hcs-viewer.spec` reads two environment variables. Unset, it behaves exactly as it does
today (ad-hoc, no entitlements). Set, it signs properly. No edit to any tracked file is needed.

| variable | effect |
|---|---|
| `SQUIDXPLORER_CODESIGN_IDENTITY` | the Developer ID passed to `codesign`; also switches the entitlements file on |
| `SQUIDXPLORER_ENTITLEMENTS` | override the default `scripts/entitlements.plist` |

`scripts/entitlements.plist` exists and is commented. It has never been used in a real signing run,
and it lists three entitlements you should try to **delete** before you trust them — read its
header, the least-privilege procedure is in there.

## What you must do personally

This needs your Apple Developer Program membership ($99/yr). Nobody can do it without your account,
and no part of it should be worked around.

**1. Get a Developer ID Application certificate.** In Xcode: Settings → Accounts → your Apple ID →
Manage Certificates → **+** → *Developer ID Application*. Confirm it landed:

```
security find-identity -v -p codesigning
```

You want the line that reads `Developer ID Application: <Name> (TEAMID)`. A *Mac Developer* or
*Apple Development* certificate is **not** the same thing and will not notarise.

**2. Store an app-specific password for notarytool.** Create one at appleid.apple.com (Sign-In and
Security → App-Specific Passwords), then:

```
xcrun notarytool store-credentials squidxplorer-notary \
  --apple-id "you@cephla.com" --team-id "TEAMID" --password "abcd-efgh-ijkl-mnop"
```

**3. Build signed, with the Hardened Runtime.** Notarisation *requires* the Hardened Runtime.

```
export SQUIDXPLORER_CODESIGN_IDENTITY="Developer ID Application: <Name> (TEAMID)"
python scripts/build_app.py --dataset /path/to/an/acquisition
```

Then re-sign the assembled bundle with `--options runtime`, because PyInstaller does not add that
flag itself:

```
codesign --force --deep --timestamp --options runtime \
  --entitlements scripts/entitlements.plist \
  --sign "$SQUIDXPLORER_CODESIGN_IDENTITY" dist/hcs-viewer.app
```

**4. Notarise and staple.**

```
ditto -c -k --keepParent dist/hcs-viewer.app /tmp/hcs-viewer.zip
xcrun notarytool submit /tmp/hcs-viewer.zip --keychain-profile squidxplorer-notary --wait
xcrun stapler staple dist/hcs-viewer.app
```

If it is rejected, `xcrun notarytool log <submission-id> --keychain-profile squidxplorer-notary` gives
the per-binary reason. The usual cause for a PyInstaller bundle is a nested `.so` that did not get
the Hardened Runtime flag; `--deep` in step 3 is what handles that.

**5. Prove it.** On a Mac that has never seen the app, after downloading it through a browser:

```
spctl -a -vv /Applications/hcs-viewer.app     # must say: accepted, source=Notarized Developer ID
```

`accepted` is the finish line. Anything else and the recipient still gets the warning.

## Ship a DMG, not a folder

`ditto`/zip is fine for notarytool, but hand customers a DMG: it preserves the signature, gives them
a drag-to-Applications target, and is what their IT expects. Notarise and staple the **DMG** as well
as the `.app` — the staple on the inner app does not cover the container.

## Windows and Linux

Not covered here and not started. Windows SmartScreen wants an Authenticode certificate (an OV
cert warns until it builds reputation; an EV cert does not). Linux AppImage has no equivalent
gate. Neither is on the critical path for the macOS demo.
