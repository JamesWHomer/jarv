# Demo animations

Terminal recordings for the main README, made with [VHS](https://github.com/charmbracelet/vhs)
as **lossless** animated WebP. This folder is intentionally **not committed** (see
`.git/info/exclude`) — the files are published as release assets on the `readme-assets`
tag and embedded from there, with a tracking issue serving as the gallery.

Style: GitHub-dark theme (`#212830` background), Berkeley Mono, FontSize 20 at
1600×868 (`update` is 428 tall), `Padding 16`.

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
│                       #   wrapper that re-records just that tape
├── output/             # rendered .webp files (created when you record)
├── bin/                # ffmpeg shim (source + compiled) for lossless WebP
├── _record-common.ps1  # shared setup (PATH/shim, effort map, retime), dot-sourced
├── record.ps1          # record one or more named tapes, sequentially
├── record-all.ps1      # re-record everything (or a subset by name), in parallel
├── retime.py           # rescale frame delays so playback isn't too fast
└── publish.ps1         # upload to the readme-assets release (URLs stay stable)
```

## Regenerating after UI changes

```powershell
.\record-all.ps1              # re-record everything, in parallel
.\record-all.ps1 hero usage   # re-record specific tapes, in parallel
.\record.ps1 hero             # re-record one (or more) tapes, sequentially
.\tapes\hero.ps1              # same, via the per-tape wrapper
.\publish.ps1                 # re-upload; URLs don't change, README needs no edit
```

`record.ps1` and `record-all.ps1` share their setup (PATH/shim bootstrap, the
per-tape reasoning-effort map, and the retime step) via `_record-common.ps1`.
Use `record.ps1` / the `tapes\<name>.ps1` wrappers to iterate on a single
animation; use `record-all.ps1` to rebuild the whole set fast.

Requirements: `vhs`, `ttyd`, `ffmpeg` (all installable via winget: `charmbracelet.vhs`,
`tsl0922.ttyd`, `Gyan.FFmpeg`), `gh` authenticated, and a working jarv install with an
API key configured — the recordings make real model calls.

Recording notes:

- Tapes record **in parallel**, grouped into one wave per `reasoning_effort` value
  (`none` for the trivial oneshot/undo questions, `low` otherwise) because the effort
  lives in the shared config.json. Your setting is restored afterwards. commands.tape
  cycles the value on camera in /settings — safe because each wave/retry re-asserts it.
- The first heads-up launch after an idle stretch sometimes comes up with dead keyboard
  input. Tapes guard against it: they `Wait` for the idle splash before typing, and use
  content `Wait` patterns after each command so a dead take fails loudly on a timeout
  instead of producing a splash-only recording. Failed takes get one sequential retry.
- VHS bakes the tapes' timing into the WebP, which plays too fast. `record-all.ps1`
  finishes by rescaling every frame's delay `1.4x` slower via `retime.py` (same
  frames, same file size — only the ANMF delay fields change). The pristine fast
  capture is kept in `output/.orig/`, so `retime.py <factor>` can re-time to a
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
