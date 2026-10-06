# Demo animations

Terminal recordings for the main README, made with [VHS](https://github.com/charmbracelet/vhs)
as **lossless** animated WebP. The rendered files are **not committed** (`output/` is
gitignored) — they are published as release assets on the `readme-assets` tag and
embedded from there, with a tracking issue serving as the gallery.

Style: GitHub-dark theme (`#212830` background), Berkeley Mono with Consolas/monospace
fallbacks, FontSize 20 at 1600×868 (`update` is 240 tall), `Padding 16` — all of it in `tapes/_common.tape`,
which every tape pulls in with `Source` and may override afterwards.

Lossless how: VHS passes ffmpeg no codec options, so its native `.webp` output is
lossy VP8 with 4:2:0 chroma subsampling, which smears colored text. `record-all.ps1`
compiles `bin\ffmpeg-shim.cs` (with the stock .NET Framework `csc.exe`) and puts the
shim ahead of the real ffmpeg on PATH; the shim appends
`-c:v libwebp_anim -lossless 1` to `.webp` encodes (exact RGB, and the animation
encoder still frame-diffs, so files stay small) and passes every other ffmpeg call
through untouched.

## Layout

```
demos/
├── tapes/              # one .tape script per animation, plus a <name>.ps1
│                       #   wrapper that re-records just that tape;
│                       #   _common.tape holds the shared Set commands
├── output/             # rendered .webp files (created when you record)
├── bin/                # tracked ffmpeg/ttyd shim sources; local binaries ignored
├── _record-common.ps1  # shared setup (PATH/shim, effort, retime), dot-sourced
├── record.ps1          # record one or more named tapes, sequentially
├── record-all.ps1      # re-record everything (or a subset by name), in parallel
├── retime.py           # rescale frame delays so playback isn't too fast
└── publish.ps1         # upload to the readme-assets release (URLs stay stable)
```

## Regenerating after UI changes

From the repository root:

```powershell
.\demos\record-all.ps1              # re-record everything, in parallel
.\demos\record-all.ps1 hero usage   # re-record specific tapes, in parallel
.\demos\record.ps1 hero             # re-record one (or more) tapes, sequentially
.\demos\tapes\hero.ps1              # same, via the per-tape wrapper
.\demos\publish.ps1                 # re-upload; URLs don't change, README needs no edit
```

`record.ps1` and `record-all.ps1` share their setup (PATH/shim bootstrap, the
recording reasoning effort, and the retime step) via `_record-common.ps1`.
Use `record.ps1` / the `tapes\<name>.ps1` wrappers to iterate on a single
animation; use `record-all.ps1` to rebuild the whole set fast.

The scripts target Windows PowerShell and compile their shims with the .NET
Framework compiler at `%WINDIR%\Microsoft.NET\Framework64\v4.0.30319\csc.exe`.
They require `vhs`, `ttyd`, `ffmpeg`, `python`, and `jarv` on PATH, plus a configured
Jarv provider/model that accepts `reasoning_effort low`. Recordings make real model
calls, save sessions/usage, and temporarily change the shared reasoning-effort
setting. The `update` tape runs `jarv /update` against that installation. An
authenticated `gh` CLI is needed for publishing and for the optional download below,
not for recording itself. VHS also needs a Chromium-family browser for capture.

**ttyd on Windows needs two workarounds**, both already wired into
`_record-common.ps1`. Symptoms are identical either way: VHS prints the tape and then
hangs forever with no error, waiting for an xterm canvas that never renders.

1. *Version.* The `ttyd.win32.exe` builds — 1.7.5 through 1.7.7, i.e. everything winget
   installs — exit the instant the browser opens the websocket on Windows 11 26xxx. The
   last build that works is 1.7.2's `ttyd.win10.exe`. Put it in `bin\ttyd\` (gitignored,
   like the shims) and the record scripts prefer it over whatever is on PATH:

   Run from the repository root:

   ```powershell
   gh release download 1.7.2 --repo tsl0922/ttyd --pattern ttyd.win10.exe --dir demos\bin\ttyd
   Move-Item demos\bin\ttyd\ttyd.win10.exe demos\bin\ttyd\ttyd.exe
   ```

2. *`--once`.* VHS always starts ttyd with `--once`, and its browser opens the page twice
   during setup, so ttyd quits on the first disconnect before the terminal is ready.
   `bin\ttyd-shim.cs` strips the flag (and job-objects the child so it still dies with
   each take), exactly like the ffmpeg shim.

Recording notes:

- `record-all.ps1` records tapes **in parallel**; `record.ps1` and the per-tape
  wrappers record sequentially. Both request `reasoning_effort low` and restore the
  previous setting afterwards. `commands.tape` cycles that shared setting on camera;
  heads-up sessions reload config after settings changes, so parallel takes are not
  isolated from shared config changes. Sequential recording reasserts `low` before
  each tape; parallel recording sets it before launch and before each retry.
- The first heads-up launch after an idle stretch sometimes comes up with dead keyboard
  input. Tapes guard against it: they `Wait` for the idle splash before typing, and use
  content `Wait` patterns after each command so a dead take fails loudly on a timeout
  instead of producing a splash-only recording. `record-all.ps1` gives failed takes
  one sequential retry; `record.ps1` reports failures without retrying.
- VHS bakes the tapes' timing into the WebP, which plays too fast. `record-all.ps1`
  finishes by rescaling every frame's delay `1.2x` slower via `retime.py` (same
  frames, same file size — only the ANMF delay fields change). The pristine fast
  capture is kept in `output/.orig/`, so `python demos/retime.py <factor>` (from the repository root) can re-time to a
  different speed without re-recording. Pillow misreads VP8L frame delays as 0;
  inspect real timing by parsing the ANMF chunks, not `im.info['duration']`.
- Verify frames with Pillow, not ffmpeg — ffmpeg can't decode animated WebP.
- VHS's `Output frames/` directory mode silently produces nothing on Windows — hence
  the ffmpeg shim instead of a record-then-reencode pipeline.

Because responses are nondeterministic, tapes use `Wait` patterns where possible and
generous `Sleep`s elsewhere. Eyeball every animation after recording (open `output/`
in a browser) before publishing.

## Hosting

- Binaries: assets on the `readme-assets` prerelease
  (`gh release view readme-assets`). `publish.ps1` uses `--clobber`, so the
  `releases/download/readme-assets/<name>.webp` URLs are stable across re-uploads.
- Gallery / tracking issue: https://github.com/JamesWHomer/jarv/issues/3

Tape scripts and shim sources are tracked in Git. Rendered animations, downloaded
executables, compiled shims, and logs are local artifacts excluded by `.gitignore`.
