/*
	The player's light gun in the headset (hotd2-vr): a Namco arcade gun in red plastic,
	held in the hand that shoots, with recoil, a muzzle flash and an optional aim line and
	dot. Drawn by xr_host.cpp into each eye after the game image.

	Room space here is the headset's local space relative to the game camera's origin
	(metres, y up), the same space the eye views are built in.

	This file is part of Flycast and is distributed under the GNU GPL v2 or later.
*/
#pragma once
#include <glm/glm.hpp>

namespace vr::xr
{

// The muzzle (the model's front lens), in gun space (the controller's aim pose: -z
// forward, y up, metres). Shots and the aim line start here, along -z. Must match
// gun_model.h (checked when compiling xr_gun.cpp).
constexpr glm::vec3 GunMuzzle { 0.00000f, 0.04710f, -0.14010f };

struct GunView
{
	glm::mat4 pose { 1.f };		// gun space -> room space, recoil included
	glm::vec3 restMuzzle { 0.f };	// room space, without recoil: where smoke leaves the barrel
	glm::vec3 restForward { 0.f, 0.f, -1.f };
	float trigger = 0.f;		// 0..1, how far the trigger is pulled
	double now = 0.0;			// seconds, any steady clock (smoke drifts with it)
	float sinceShot = 1e9f;		// seconds since the last shot
	unsigned shot = 0;			// counts shots: each looks a bit different
	bool aimLine = false;		// draw the line from the muzzle to the aim point
	bool aimDot = false;		// draw the dot at the aim point
	glm::vec3 aimPoint { 0.f };	// room space
};

// How hard the gun kicks back, sinceShot seconds after a shot: a sharp kick, a small
// bounce forward, settled after about a quarter second. 0..1, can go slightly negative.
float gunKick(float sinceShot);

// Draws the gun into the bound eye framebuffer, on top of the game image.
// viewProj: room space -> this eye's clip space; eyePos: the eye, in room space.
void drawGun(const glm::mat4& viewProj, const glm::vec3& eyePos, const GunView& gun);
// The GL context is going away.
void termGun();

}
