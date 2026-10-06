# The House of the Dead 2 VR

Stand inside **The House of the Dead 2** (Sega Dreamcast) on a **Meta Quest 3**, standalone.
The game's camera becomes your rail cart, the 3D scenes are rebuilt around you in stereo,
and your controller is the light gun, a red Namco-style arcade gun in your hand.

It is a fork of the [Flycast](https://github.com/flyinghead/flycast) Dreamcast emulator,
branch `hotd2-vr`. There is no remake here: the original game runs, and the emulator turns
what it draws into a VR view.

> **No game included.** You need your own dump of the game. This project contains no
> Sega code, data, graphics or sound. Not affiliated with or endorsed by Sega, Namco,
> Meta or the Flycast project.

Status: experimental, built for and tested with the European release (PAL, product ID
`MK-5100250`) on a Quest 3. Other versions need their own profile (see *How it works*).
The newest parts, widening the game's own field of view and the fire animation, have
not been tried on the headset yet; `vr.WidenFov=no` goes back to the earlier, tested
widening.

## What you get

- **Immersive 3D.** The Dreamcast's GPU only sees vertices already projected to the
  screen (x, y, 1/w). Knowing the game's lens, each one is lifted back into 3D and drawn
  per eye from your head, so you look around and lean into the scene at the headset's
  refresh rate, while the game runs at its own 60 Hz.
- **A wider view than the original.** The game's own field of view is raised from 41° to
  74° (vertical), inside the game: it draws, culls and checks your shots with it.
- **2D on a screen in front of you.** Menus, text and the HUD sit on a plane at a fixed
  distance, framed as in the original.
- **The light gun.** Either controller. Shots are traced into the rebuilt scene from the
  barrel, so you hit what you see; a laser line and dot show where (optional). Recoil,
  muzzle flash, smoke and a haptic knock on every shot.
- **Comfort.** Things that come right into your face (a zombie grabbing you) are pulled
  back a little, so your eyes don't have to cross. Recentering, adjustable world size.
- Starts straight into the game from the headset's app library.

## Controls

| | Left controller | Right controller |
|---|---|---|
| Shoot | trigger | trigger |
| Reload | Y, or shoot away from the screen | B, or shoot away from the screen |
| Start (also skips cut scenes) | menu button ≡ | A |
| Menus (the gun's D-pad) | thumbstick | thumbstick |
| Recenter | X or thumbstick click | thumbstick click |
| World size (kept) | hold grip + thumbstick up/down | hold grip + thumbstick up/down |

The gun goes to the hand you last fired with. Holding the Meta button recenters too.
Quit with the Meta button, then *Quit*.

## Build

On Windows, with:

- JDK 17 (a Microsoft JDK 17 in Program Files is picked up, otherwise set `JAVA_HOME`)
- the Android SDK with NDK 29.0.14206865 and CMake 3.22.1 (`ANDROID_HOME`, default
  `%LOCALAPPDATA%\Android\Sdk`)
- the git submodules: `git submodule update --init --recursive`

On Windows, two paths in the `core/deps/gamesdk` submodule are symlinks
(`games-frame-pacing/include/common` and `include/swappy`). Without symlink support
(Developer Mode plus `git config --global core.symlinks true` before cloning) they
check out as small text files and the build fails: replace them with copies of
`core/deps/gamesdk/include/common` and `core/deps/gamesdk/include/swappy`.

```bat
hotd2-vr\build-quest.cmd
```

This builds the `vr` build type (arm64, optimised, OpenXR) to
`shell\android-studio\flycast\build\intermediates\apk\vr\flycast-vr.apk`. It installs
as its own app, *HOTD2 VR* (`com.flycast.emulator.vr`), next to a normal Flycast.

`hotd2-vr\build-win.cmd` builds the Windows version (Visual Studio 2022 Build Tools),
for the webcam finger-gun mode and for testing.

## Install on the Quest

1. Put the headset in developer mode and connect it with `adb`.
2. Install the app:
   ```bat
   adb install -r -t flycast-vr.apk
   ```
3. Copy your game into the app's own storage (the app cannot read files that `adb push`
   puts in shared storage). Use short file names and a `.cue` that lists them:
   ```bat
   adb shell run-as com.flycast.emulator.vr mkdir -p files/games
   adb exec-in run-as com.flycast.emulator.vr sh -c "cat > files/games/track1.bin" < "Track 1.bin"
   rem ...the same for the other tracks and the .cue
   ```
   The app boots the first `.cue`, `.gdi`, `.chd` or `.cdi` in `files/games`.
4. Start *HOTD2 VR* from the library (*Unknown sources*).

No Dreamcast BIOS is needed: Flycast's built-in replacement (HLE) works.

## Options

Settings live in `/sdcard/Android/data/com.flycast.emulator.vr/files/emu.cfg`, under
`[config]`. The useful ones:

| Option | Default | |
|---|---|---|
| `vr.WorldScale` | 0.025 | metres per game unit (also set with grip + thumbstick) |
| `vr.FovScale` | 2 (profile) | how much wider than the original (tan of the half angle) |
| `vr.WidenFov` | yes | widen through the game's own field of view (the profile knows where) |
| `vr.Laser` | yes | laser line and dot |
| `vr.ShowGun` | yes | the gun model |
| `vr.ComfortStart` / `vr.ComfortMin` | 0.8 / 0.4 | comfort zone, metres (0: off) |
| `vr.HudDepth` | 40 | distance of the 2D plane, game units |
| `vr.XrRefreshRate` | 72 | Hz |
| `vr.XrResolution` | 1 | eye buffer size, times the recommended size |

## How it works

- `core/rend/vr_reproject.*`: rebuilds eye space from the projected vertices with the
  game's live focal length (read from its projection and viewport matrices in RAM), and
  the per-game profile: matrix addresses and the field-of-view literals to patch.
  HOTD2's four perspective setups load their angle (`0x1D3C`, 41.1°) from two literals
  (`0x8C029C12`, `0x8C02B370`) for the routine at `0x8C0383C0`.
- `core/rend/gles/gles.cpp`: the vertex shader does the lifting and reprojection; only
  frames the game actually presents reach the headset.
- `core/rend/vr/xr_host.*`: the OpenXR session (bound to Flycast's EGL context),
  headset-paced frames, controllers and the light gun ray cast.
- `core/rend/vr/xr_gun.*`, `gun_model.h`: the gun, its effects and shaders.
- `core/input/udp_lightgun.*`: the light gun can also be driven over UDP on 127.0.0.1
  (used by the finger-gun tracker).

## Tools

- `hotd2-vr/fingergun`: play on the PC with **finger guns** in front of a webcam
  (MediaPipe hand tracking). Set up once:
  ```bat
  py -m venv hotd2-vr\fingergun\.venv
  hotd2-vr\fingergun\.venv\Scripts\pip install -r hotd2-vr\fingergun\requirements.txt
  ```
  and put MediaPipe's [hand landmarker model](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
  (`hand_landmarker.task`) in `hotd2-vr\fingergun\`. Then `hotd2-vr\play-fingergun.ps1`
  (after `build-win.cmd`). It starts with a guided calibration; its on-screen text is in Dutch.
- `hotd2-vr/gun_model`: bakes a `.glb` gun model into `gun_model.h`.
- `hotd2-vr/re/sh4dis.py`: disassembles game code from a Flycast RAM dump (Capstone),
  which is how the field-of-view literals were found.
- `hotd2-vr/run-test.ps1`: PC test runs with screenshots.

The scripts look for your game in a `game` folder next to the repository, or in
`%HOTD2VR_DATA%\game`.

## Credits and licences

- [Flycast](https://github.com/flyinghead/flycast) by flyinghead and contributors, GPL-2.0
  or later. This fork's code is too (see `LICENSE`); changed Flycast files carry a note,
  and `git log master..hotd2-vr` shows every change. Flycast's own readme:
  [README.flycast.md](README.flycast.md).
- The gun: ["Namco Arcade Gun"](https://sketchfab.com/3d-models/namco-arcade-gun-15fbd5b9add94a34b5c21746e3dd32be)
  by [Martoscar](https://sketchfab.com/Martoscar), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changed: re-oriented and
  moved into the controller's aim space, quantised, textures and materials dropped,
  recoloured, converted to a C++ header (`core/rend/vr/gun_model.h`, CC BY 4.0). Since CC
  BY 4.0 goes together with GPL-3.0 but not GPL-2.0, builds that include it are
  distributed under GPL-3.0, which Flycast's "or later" allows.
- [OpenXR loader](https://github.com/KhronosGroup/OpenXR-SDK) by Khronos, Apache-2.0.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) by Google, Apache-2.0
  (finger-gun tracker); [Capstone](https://www.capstone-engine.org/), BSD.
- Inspired by [DR-89/time-crisis-vr](https://github.com/DR-89/time-crisis-vr).
- The House of the Dead is a trademark of Sega. Namco is a trademark of Bandai Namco.
