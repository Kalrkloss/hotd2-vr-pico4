/*
	Lightgun input over UDP (hotd2-vr). See udp_lightgun.cpp.

	This file is part of Flycast and is distributed under the GNU GPL v2 or later.
*/
#pragma once

#include "types.h"

// Aim and fire the emulated light gun directly (the UDP datagrams and the headset's
// controller both end up here). x, y in 1/10000ths of the game screen (outside is
// off-screen); buttons: bit 0 trigger, bit 1 reload (off-screen shot), bit 2 start.
void lightgunSet(int player, int x, int y, u32 buttons);

// True while the UDP light gun drives this port (a datagram arrived recently).
// Mouse and analog-stick aiming stand aside meanwhile, so they can't fight over the crosshair.
bool udpLightgunActive(int port);
