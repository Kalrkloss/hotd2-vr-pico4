/*
	VR reprojection prototype (hotd2-vr). See vr_reproject.h.

	This file is part of Flycast and is distributed under the GNU GPL v2 or later.
*/
#include "vr_reproject.h"
#include "vr/xr_host.h"
#include "transform_matrix.h"
#include "hw/pvr/ta_ctx.h"
#include "hw/sh4/sh4_mem.h"
#include "hw/maple/maple_cfg.h"
#include "cfg/option.h"
#include "emulator.h"
#include "log/Log.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <cstring>
#include <vector>

namespace vr
{

static ReprojectParams params;
static u64 frameCount;

// Near plane as a fraction of the closest vertex depth: vertices are never clipped by it
// in the identity reprojection, but anything behind the free camera is.
constexpr float NearFraction = 0.01f;

// Offsets of the x and y scale terms in a 4x4 float matrix (diagonal, either order).
constexpr u32 ScaleX = 0;
constexpr u32 ScaleY = 5 * 4;

static float readFloat(u32 ramOffset)
{
	u32 raw = ReadMem32_nommu(0x8C000000u + (ramOffset & 0x00FFFFFFu));
	float f;
	memcpy(&f, &raw, sizeof(f));
	return f;
}

static void writeFloat(u32 ramOffset, float f)
{
	u32 raw;
	memcpy(&raw, &f, sizeof(raw));
	WriteMem32_nommu(0x8C000000u + (ramOffset & 0x00FFFFFFu), raw);
}

static bool plausible(float v, float lo, float hi) {
	return std::isfinite(v) && std::abs(v) > lo && std::abs(v) < hi;
}

// The game's focal length in DC pixels, per axis: viewport scale times projection scale.
static glm::vec2 readGameFocal()
{
	if (config::VrProjAddr > 0 && config::VrViewportAddr > 0)
	{
		const glm::vec2 focal(
				std::abs(readFloat(config::VrViewportAddr + ScaleX) * readFloat(config::VrProjAddr + ScaleX)),
				std::abs(readFloat(config::VrViewportAddr + ScaleY) * readFloat(config::VrProjAddr + ScaleY)));
		if (plausible(focal.x, 10.f, 20000.f) && plausible(focal.y, 10.f, 20000.f))
			return focal;
	}
	return glm::vec2(config::VrFocal);
}

//
// Widening the game's view, two ways.
//
// Best: the field of view the game builds its projection from. HOTD2 passes it as a
// 16-bit angle (65536 = a full turn) from a few literals in its code; raised there, the
// game itself draws, culls and tests light gun hits with the wider view. Checked every
// vblank, since the code is loaded after the game starts. See GameProfile::fovLiterals.
//
// Otherwise: scale down the viewport matrix's x/y terms by vr.FovScale. The game then
// maps (and culls against the screen) a wider cone into the same 640x480, but anything
// it computes for the screen on its own, like the light gun hit test, keeps the stock
// view. The projection itself can't be scaled: the game rebuilds it every frame. The
// viewport is set once per scene, so it is checked every vblank: a value differing from
// what we last wrote is fresh from the game and gets scaled; our own value is left alone.
//
static glm::vec2 lastWritten;
static u32 fovLiterals[4];		// RAM offsets of the game's field of view angle (0: none)
static u16 fovStock;			// the angle the game ships with

// The angle (65536 = full turn) that widens stock by `scale` (as tan of the half angle).
static u16 widenedAngle(u16 stock, float scale)
{
	const float half = stock * (3.14159265f / 65536.f);
	const float wide = 2.f * std::atan(std::tan(half) * scale);
	return (u16)std::lround(wide * (65536.f / 6.2831853f));
}

bool widensGameFov() {
	return fovLiterals[0] != 0;
}

static void widenGameView(Event event, void *)
{
	if (event == Event::Start)
	{
		lastWritten = glm::vec2(0.f);
		return;
	}
	const float scale = config::VrFovScale;
	if (!config::VrReproject || scale <= 1.f)
		return;
	if (widensGameFov())
	{
		const u16 wide = widenedAngle(fovStock, scale);
		for (u32 offset : fovLiterals)
		{
			if (offset == 0)
				break;
			const u32 addr = 0x8C000000u + offset;
			// only the stock value: anything else isn't the code we know (not loaded yet)
			if (ReadMem16_nommu(addr) == fovStock)
			{
				WriteMem16_nommu(addr, wide);
				NOTICE_LOG(RENDERER, "VR: field of view %.1f -> %.1f degrees at %08x", fovStock * 360.f / 65536.f, wide * 360.f / 65536.f, addr);
			}
		}
		return;
	}
	if (config::VrViewportAddr <= 0)
		return;
	const u32 addr = config::VrViewportAddr;
	const glm::vec2 cur(readFloat(addr + ScaleX), readFloat(addr + ScaleY));
	if (cur == lastWritten || !plausible(cur.x, 10.f, 5000.f) || !plausible(cur.y, 10.f, 5000.f))
		return;
	lastWritten = cur / scale;
	writeFloat(addr + ScaleX, lastWritten.x);
	writeFloat(addr + ScaleY, lastWritten.y);
}

static struct WidenGameViewRegistration
{
	WidenGameViewRegistration() {
		EventManager::listen(Event::Start, widenGameView);
		EventManager::listen(Event::VBlank, widenGameView);
	}
} widenGameViewRegistration;

static void logSceneStats(const rend_context& ctx, glm::vec2 focal)
{
	std::vector<float> invW;
	invW.reserve(ctx.verts.size());
	size_t overlay = 0;
	for (const Vertex& v : ctx.verts)
		if (std::isfinite(v.z) && v.z > 0.f)
		{
			invW.push_back(v.z);
			if (std::abs(1.f / v.z - config::VrHudW) < 0.002f)
				overlay++;
		}
	if (invW.empty())
		return;
	std::sort(invW.begin(), invW.end());
	auto w = [&](float q) { return 1.f / invW[(size_t)(q * (invW.size() - 1))]; };
	NOTICE_LOG(RENDERER, "VR: focal %.1f x %.1f  verts %zu overlay %zu  W near %.3f p10 %.3f median %.3f p90 %.3f far %.3f",
			focal.x, focal.y, invW.size(), overlay,
			w(1.f), w(0.9f), w(0.5f), w(0.1f), w(0.f));
}

//
// Per-game camera addresses, applied when the game starts so the Quest build needs no
// manual config. Values already set in the config file win.
//
struct GameProfile
{
	const char *gameId;
	int projAddr, viewportAddr;
	float fovScale;
	// Where the game's code holds its field of view (16-bit angle literals), and its value.
	u32 fovLiterals[4];
	u16 fovStock;
};
static const GameProfile profiles[] = {
	// The House of the Dead 2 (PAL). Its four perspective setups (0x8C029B48, 0x8C02B2C6,
	// 0x8C02B30A, 0x8C02B34E) load 41.1 degrees from two literals and call 0x8C0383C0.
	{ "MK-5100250", 0x4C65E0, 0x4C6708, 2.f, { 0x029C12, 0x02B370 }, 7484 },
};

static void applyGameProfile(Event, void *)
{
	std::fill(std::begin(fovLiterals), std::end(fovLiterals), 0u);
	if (xr::enabled())
	{
		// Everything the headset view depends on.
		config::VrReproject.override(true);
		config::NativeDepthInterpolation.override(true);
	}
	for (const GameProfile& prof : profiles)
	{
		if (settings.content.gameId != prof.gameId)
			continue;
		if (config::VrProjAddr == 0)
			config::VrProjAddr.override(prof.projAddr);
		if (config::VrViewportAddr == 0)
			config::VrViewportAddr.override(prof.viewportAddr);
		if (xr::enabled() && config::VrWiden && config::VrFovScale <= 1.f)
			config::VrFovScale.override(prof.fovScale);
		if (config::VrWidenFov)
		{
			std::copy(std::begin(prof.fovLiterals), std::end(prof.fovLiterals), std::begin(fovLiterals));
			fovStock = prof.fovStock;
		}
		if (xr::enabled() && config::VrXrGun && config::MapleMainDevices[0] != MDT_LightGun)
		{
			// The right controller is a light gun: plug one into port A (a pad ignores
			// where it points). The devices were made just before this event, before the
			// game runs, so they can still be swapped.
			config::MapleMainDevices[0].override(MDT_LightGun);
			mcfg_DestroyDevices();
			mcfg_CreateDevices();
			NOTICE_LOG(RENDERER, "VR: light gun in port A");
		}
		NOTICE_LOG(RENDERER, "VR: camera profile for %s", prof.gameId);
	}
}

static struct GameProfileRegistration
{
	GameProfileRegistration() {
		EventManager::listen(Event::Start, applyGameProfile);
	}
} gameProfileRegistration;

GameCamera gameCamera(const rend_context& ctx)
{
	int dcWidth, dcHeight;
	getPvrFramebufferSize(ctx, dcWidth, dcHeight);
	GameCamera cam;
	cam.dcSize = glm::vec2(dcWidth, dcHeight);
	cam.tanHalf = cam.dcSize * 0.5f / readGameFocal();
	cam.overlayW = config::VrHudW;
	return cam;
}

static glm::mat4 projection(glm::vec2 tanHalf, float zNear)
{
	glm::mat4 proj(0.f);
	proj[0][0] = 1.f / tanHalf.x;
	proj[1][1] = 1.f / tanHalf.y;
	proj[2][2] = -1.f;
	proj[2][3] = -1.f;
	proj[3][2] = -2.f * zNear;
	return proj;
}

glm::vec3 comfortParams()
{
	const float start = config::VrComfortStart;
	if (start <= 0.f)
		return glm::vec3(0.f);
	// at most half the start distance: the curve stays monotonic
	const float closest = std::clamp((float)config::VrComfortMin, 0.f, start * 0.5f);
	return glm::vec3((float)config::VrWorldScale, start, closest);
}

const ReprojectParams& update(const rend_context& ctx)
{
	int dcWidth, dcHeight;
	getPvrFramebufferSize(ctx, dcWidth, dcHeight);
	const glm::vec2 halfSize(dcWidth * 0.5f, dcHeight * 0.5f);
	const bool active = config::VrReproject && !ctx.isRTT;
	params.comfort = glm::vec3(0.f);
	params.shot = glm::vec4(0.f);

	const float zMax = ctx.fZ_max > 0.f && std::isfinite(ctx.fZ_max) ? ctx.fZ_max : 1.f;
	const float zNear = NearFraction / zMax;

	if (!active)
	{
		params.tanHalf = halfSize / (float)config::VrFocal;
		params.overlay = glm::vec4(params.tanHalf, 1.f, -1.f);	// no vertex counts as overlay
		params.viewProj = projection(params.tanHalf, zNear);
		return params;
	}

	// The game may run with a widened view: rebuild with its live focal length, but
	// present with the stock one so the framing and HUD size stay as designed.
	const glm::vec2 gameFocal = readGameFocal();
	const glm::vec2 stockTan = halfSize / (float)config::VrFocal;
	params.tanHalf = halfSize / gameFocal;
	params.overlay = glm::vec4(stockTan, (float)config::VrHudDepth, (float)config::VrHudW);

	// depth statistics now and then, for tuning a new game on the PC (not in the headset)
	if (frameCount++ % 120 == 0 && !xr::enabled())
		logSceneStats(ctx, gameFocal);

	if (const xr::Eye *eye = xr::currentEye())
	{
		// In the headset: game camera at the player's head, viewed through this eye.
		params.viewProj = eye->viewProj;
		params.comfort = comfortParams();
		glm::vec2 shot;
		if (widensGameFov() && xr::recentShot(shot))
			params.shot = glm::vec4(shot * 2.f - 1.f, 0.2f, 1.f);
		return params;
	}

	float yaw = config::VrYaw;
	float pitch = config::VrPitch;
	glm::vec3 offset(config::VrOffsetX, config::VrOffsetY, config::VrOffsetZ);
	if (config::VrAnimate)
	{
		// Sway around the game camera so stills from consecutive seconds show parallax.
		const float t = frameCount / 60.f;
		yaw *= std::sin(t * 0.9f);
		pitch *= std::sin(t * 0.6f);
		offset *= glm::vec3(std::sin(t * 0.7f), std::cos(t * 0.5f), std::sin(t * 0.4f));
	}
	glm::mat4 pose = glm::translate(glm::mat4(1.f), offset)
			* glm::rotate(glm::mat4(1.f), glm::radians(yaw), glm::vec3(0, 1, 0))
			* glm::rotate(glm::mat4(1.f), glm::radians(pitch), glm::vec3(1, 0, 0));
	// The renderer's ndc y points down the image; flip into y-up space for the pose.
	const glm::mat4 flipY = glm::scale(glm::mat4(1.f), glm::vec3(1, -1, 1));
	params.viewProj = projection(stockTan, zNear) * flipY * glm::inverse(pose) * flipY;

	return params;
}

}
