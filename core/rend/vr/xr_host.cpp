/*
	OpenXR host (hotd2-vr). See xr_host.h.

	Everything here runs on Flycast's render thread, which owns the EGL context the
	OpenXR session is bound to. Initialisation happens on the first frame.

	World layout: the game camera sits at the player's head as it was when the session
	started (or at the last recenter), looking where the player looked. Rebuilt game
	eye-space positions (game units, renderer y convention: y down) map to the headset's
	local space (metres, y up) by a uniform scale (vr.WorldScale) and a y flip.

	This file is part of Flycast and is distributed under the GNU GPL v2 or later.
*/
#include "xr_host.h"

#ifdef USE_OPENXR
#include "xr_gun.h"
#include <jni.h>
#include <glad/egl.h>
#include "rend/gles/gles.h"
#define XR_USE_PLATFORM_ANDROID
#define XR_USE_GRAPHICS_API_OPENGL_ES
#include <openxr/openxr.h>
#include <openxr/openxr_platform.h>

#include "rend/vr_reproject.h"
#include "rend/TexCache.h"
#include "hw/pvr/Renderer_if.h"
#include "hw/pvr/ta_ctx.h"
#include "input/udp_lightgun.h"
#include "cfg/option.h"
#include "cfg/cfg.h"
#include "log/Log.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>
#include <cstring>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

extern JavaVM *g_jvm;
extern jobject g_activity;

namespace vr::xr
{
namespace
{

constexpr float NearMetres = 0.05f;
constexpr u32 RestartIndex = 0xFFFFFFFFu;

XrInstance instance = XR_NULL_HANDLE;
XrSystemId systemId = XR_NULL_SYSTEM_ID;
XrSession session = XR_NULL_HANDLE;
XrSpace localSpace = XR_NULL_HANDLE;
// Either controller can be the gun: the one whose trigger was pulled last.
constexpr int Left = 0, Right = 1;
XrPath handPaths[2];
XrSpace aimSpaces[2];
int gunHand = Right;
XrActionSet actionSet = XR_NULL_HANDLE;
XrAction aimAction, triggerAction, startAction, reloadAction, recenterAction, hapticAction, stickAction, gripAction;
PFN_xrRequestDisplayRefreshRateFB requestRefreshRate;
PFN_xrPerfSettingsSetPerformanceLevelEXT setPerformanceLevel;

bool initTried;
bool sessionRunning;
bool focused;
std::chrono::steady_clock::time_point focusLost;
// The runtime lost the session (start a new one) or wants the app to quit (don't).
bool sessionLost;
bool exiting;

struct EyeChain
{
	XrSwapchain chain = XR_NULL_HANDLE;
	int width = 0, height = 0;
	std::vector<XrSwapchainImageOpenGLESKHR> images;
	std::vector<GLuint> fbos;
	GLuint depthStencil = 0;
};
EyeChain chains[2];
XrView views[2] { { XR_TYPE_VIEW }, { XR_TYPE_VIEW } };

Eye eye;
bool drawingEye;

// Where the game camera is: the head pose at start / recenter (yaw only).
bool originSet;
bool recenterRequested;
glm::vec3 originPos;
glm::quat originRot { 1, 0, 0, 0 };

// Light gun state
bool triggerWas, recenterWas;
// The pistol as drawn: where it is (no pose: hidden), and the last shot for recoil/flash.
GunView gunView;
bool gunVisible;
XrTime lastShot;

bool check(XrResult result, const char *what)
{
	if (XR_SUCCEEDED(result))
		return true;
	char text[XR_MAX_RESULT_STRING_SIZE] = "";
	if (instance != XR_NULL_HANDLE)
		xrResultToString(instance, result, text);
	ERROR_LOG(RENDERER, "XR: %s failed: %d %s", what, result, text);
	return false;
}
#define XRCHECK(call) check((call), #call)

XrPath path(const char *s)
{
	XrPath p = XR_NULL_PATH;
	xrStringToPath(instance, s, &p);
	return p;
}

bool createAction(XrAction& action, const char *name, const char *label, XrActionType type, bool perHand = false)
{
	XrActionCreateInfo info { XR_TYPE_ACTION_CREATE_INFO };
	info.actionType = type;
	strncpy(info.actionName, name, sizeof(info.actionName) - 1);
	strncpy(info.localizedActionName, label, sizeof(info.localizedActionName) - 1);
	if (perHand)
	{
		info.countSubactionPaths = 2;
		info.subactionPaths = handPaths;
	}
	return XRCHECK(xrCreateAction(actionSet, &info, &action));
}

bool initActions()
{
	handPaths[Left] = path("/user/hand/left");
	handPaths[Right] = path("/user/hand/right");
	XrActionSetCreateInfo setInfo { XR_TYPE_ACTION_SET_CREATE_INFO };
	strcpy(setInfo.actionSetName, "lightgun");
	strcpy(setInfo.localizedActionSetName, "Light gun");
	if (!XRCHECK(xrCreateActionSet(instance, &setInfo, &actionSet)))
		return false;
	if (!createAction(aimAction, "aim", "Aim", XR_ACTION_TYPE_POSE_INPUT, true)
			|| !createAction(triggerAction, "trigger", "Fire", XR_ACTION_TYPE_FLOAT_INPUT, true)
			|| !createAction(startAction, "start", "Start", XR_ACTION_TYPE_BOOLEAN_INPUT)
			|| !createAction(reloadAction, "reload", "Reload", XR_ACTION_TYPE_BOOLEAN_INPUT)
			|| !createAction(recenterAction, "recenter", "Recenter", XR_ACTION_TYPE_BOOLEAN_INPUT)
			|| !createAction(hapticAction, "recoil", "Recoil", XR_ACTION_TYPE_VIBRATION_OUTPUT, true)
			|| !createAction(stickAction, "stick", "Menus (D-pad), world size", XR_ACTION_TYPE_VECTOR2F_INPUT)
			|| !createAction(gripAction, "grip", "Hold for world size", XR_ACTION_TYPE_FLOAT_INPUT))
		return false;
	// Either trigger fires. Start (also skips cut scenes): A, or the menu button on the left.
	// Reload: B or Y (or shooting away from the screen). Recenter: X, or a thumbstick click.
	// Thumbstick: the gun's D-pad (menus); with a grip held, up/down sizes the world.
	const XrActionSuggestedBinding bindings[] {
		{ stickAction, path("/user/hand/left/input/thumbstick") },
		{ stickAction, path("/user/hand/right/input/thumbstick") },
		{ gripAction, path("/user/hand/left/input/squeeze/value") },
		{ gripAction, path("/user/hand/right/input/squeeze/value") },
		{ aimAction, path("/user/hand/left/input/aim/pose") },
		{ aimAction, path("/user/hand/right/input/aim/pose") },
		{ triggerAction, path("/user/hand/left/input/trigger/value") },
		{ triggerAction, path("/user/hand/right/input/trigger/value") },
		{ startAction, path("/user/hand/right/input/a/click") },
		{ startAction, path("/user/hand/left/input/menu/click") },
		{ reloadAction, path("/user/hand/right/input/b/click") },
		{ reloadAction, path("/user/hand/left/input/y/click") },
		{ recenterAction, path("/user/hand/left/input/x/click") },
		{ recenterAction, path("/user/hand/left/input/thumbstick/click") },
		{ recenterAction, path("/user/hand/right/input/thumbstick/click") },
		{ hapticAction, path("/user/hand/left/output/haptic") },
		{ hapticAction, path("/user/hand/right/output/haptic") },
	};
	XrInteractionProfileSuggestedBinding suggested { XR_TYPE_INTERACTION_PROFILE_SUGGESTED_BINDING };
	suggested.interactionProfile = path("/interaction_profiles/oculus/touch_controller");
	suggested.countSuggestedBindings = (u32)std::size(bindings);
	suggested.suggestedBindings = bindings;
	if (!XRCHECK(xrSuggestInteractionProfileBindings(instance, &suggested)))
		return false;
	XrSessionActionSetsAttachInfo attach { XR_TYPE_SESSION_ACTION_SETS_ATTACH_INFO };
	attach.countActionSets = 1;
	attach.actionSets = &actionSet;
	if (!XRCHECK(xrAttachSessionActionSets(session, &attach)))
		return false;
	for (int hand : { Left, Right })
	{
		XrActionSpaceCreateInfo spaceInfo { XR_TYPE_ACTION_SPACE_CREATE_INFO };
		spaceInfo.action = aimAction;
		spaceInfo.subactionPath = handPaths[hand];
		spaceInfo.poseInActionSpace.orientation.w = 1.f;
		if (!XRCHECK(xrCreateActionSpace(session, &spaceInfo, &aimSpaces[hand])))
			return false;
	}
	return true;
}

bool initSwapchains()
{
	u32 formatCount = 0;
	if (!XRCHECK(xrEnumerateSwapchainFormats(session, 0, &formatCount, nullptr)))
		return false;
	std::vector<int64_t> formats(formatCount);
	if (!XRCHECK(xrEnumerateSwapchainFormats(session, formatCount, &formatCount, formats.data())))
		return false;
	// Plain RGBA8: the game's colours were made for a gamma display, like the eye buffers.
	if (std::find(formats.begin(), formats.end(), (int64_t)GL_RGBA8) == formats.end())
	{
		ERROR_LOG(RENDERER, "XR: no RGBA8 swapchain format");
		return false;
	}
	u32 viewCount = 0;
	if (!XRCHECK(xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 0, &viewCount, nullptr))
			|| viewCount != 2)
		return false;
	XrViewConfigurationView configViews[2] { { XR_TYPE_VIEW_CONFIGURATION_VIEW }, { XR_TYPE_VIEW_CONFIGURATION_VIEW } };
	if (!XRCHECK(xrEnumerateViewConfigurationViews(instance, systemId, XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO, 2, &viewCount, configViews)))
		return false;
	for (int i = 0; i < 2; i++)
	{
		EyeChain& c = chains[i];
		const float scale = std::clamp((float)config::VrXrResolution, 0.3f, 1.5f);
		c.width = (int)(configViews[i].recommendedImageRectWidth * scale);
		c.height = (int)(configViews[i].recommendedImageRectHeight * scale);
		XrSwapchainCreateInfo info { XR_TYPE_SWAPCHAIN_CREATE_INFO };
		info.usageFlags = XR_SWAPCHAIN_USAGE_COLOR_ATTACHMENT_BIT | XR_SWAPCHAIN_USAGE_SAMPLED_BIT;
		info.format = GL_RGBA8;
		info.sampleCount = 1;
		info.width = c.width;
		info.height = c.height;
		info.faceCount = 1;
		info.arraySize = 1;
		info.mipCount = 1;
		if (!XRCHECK(xrCreateSwapchain(session, &info, &c.chain)))
			return false;
		u32 imageCount = 0;
		if (!XRCHECK(xrEnumerateSwapchainImages(c.chain, 0, &imageCount, nullptr)))
			return false;
		c.images.assign(imageCount, { XR_TYPE_SWAPCHAIN_IMAGE_OPENGL_ES_KHR });
		if (!XRCHECK(xrEnumerateSwapchainImages(c.chain, imageCount, &imageCount, (XrSwapchainImageBaseHeader *)c.images.data())))
			return false;
		// Depth with stencil: the renderer uses stencil for modifier volumes (shadows).
		glGenRenderbuffers(1, &c.depthStencil);
		glBindRenderbuffer(GL_RENDERBUFFER, c.depthStencil);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, c.width, c.height);
		c.fbos.resize(imageCount);
		glGenFramebuffers(imageCount, c.fbos.data());
		for (u32 j = 0; j < imageCount; j++)
		{
			glBindFramebuffer(GL_FRAMEBUFFER, c.fbos[j]);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, c.images[j].image, 0);
			glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, c.depthStencil);
			if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
			{
				ERROR_LOG(RENDERER, "XR: eye framebuffer incomplete");
				return false;
			}
		}
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	NOTICE_LOG(RENDERER, "XR: eye buffers %dx%d", chains[0].width, chains[0].height);
	return true;
}

bool init()
{
	initTried = true;
	if (g_jvm == nullptr || g_activity == nullptr)
	{
		ERROR_LOG(RENDERER, "XR: no Java VM / activity");
		return false;
	}
	PFN_xrInitializeLoaderKHR initLoader = nullptr;
	if (!XRCHECK(xrGetInstanceProcAddr(XR_NULL_HANDLE, "xrInitializeLoaderKHR", (PFN_xrVoidFunction *)&initLoader)))
		return false;
	XrLoaderInitInfoAndroidKHR loaderInfo { XR_TYPE_LOADER_INIT_INFO_ANDROID_KHR };
	loaderInfo.applicationVM = g_jvm;
	loaderInfo.applicationContext = g_activity;
	if (!XRCHECK(initLoader((const XrLoaderInitInfoBaseHeaderKHR *)&loaderInfo)))
		return false;

	std::vector<const char *> extensions { XR_KHR_ANDROID_CREATE_INSTANCE_EXTENSION_NAME, XR_KHR_OPENGL_ES_ENABLE_EXTENSION_NAME };
	u32 available = 0;
	XRCHECK(xrEnumerateInstanceExtensionProperties(nullptr, 0, &available, nullptr));
	std::vector<XrExtensionProperties> props(available, { XR_TYPE_EXTENSION_PROPERTIES });
	XRCHECK(xrEnumerateInstanceExtensionProperties(nullptr, available, &available, props.data()));
	auto has = [&](const char *name) {
		for (const auto& p : props)
			if (!strcmp(p.extensionName, name))
				return true;
		return false;
	};
	const bool refreshRates = has(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
	const bool perfSettings = has(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME);
	if (refreshRates)
		extensions.push_back(XR_FB_DISPLAY_REFRESH_RATE_EXTENSION_NAME);
	if (perfSettings)
		extensions.push_back(XR_EXT_PERFORMANCE_SETTINGS_EXTENSION_NAME);

	XrInstanceCreateInfoAndroidKHR androidInfo { XR_TYPE_INSTANCE_CREATE_INFO_ANDROID_KHR };
	androidInfo.applicationVM = g_jvm;
	androidInfo.applicationActivity = g_activity;
	XrInstanceCreateInfo createInfo { XR_TYPE_INSTANCE_CREATE_INFO };
	createInfo.next = &androidInfo;
	createInfo.enabledExtensionCount = (u32)extensions.size();
	createInfo.enabledExtensionNames = extensions.data();
	strcpy(createInfo.applicationInfo.applicationName, "HOTD2 VR");
	createInfo.applicationInfo.applicationVersion = 1;
	strcpy(createInfo.applicationInfo.engineName, "Flycast");
	createInfo.applicationInfo.engineVersion = 1;
	createInfo.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 0, 34);
	if (!XRCHECK(xrCreateInstance(&createInfo, &instance)))
		return false;
	if (refreshRates)
		xrGetInstanceProcAddr(instance, "xrRequestDisplayRefreshRateFB", (PFN_xrVoidFunction *)&requestRefreshRate);
	if (perfSettings)
		xrGetInstanceProcAddr(instance, "xrPerfSettingsSetPerformanceLevelEXT", (PFN_xrVoidFunction *)&setPerformanceLevel);

	XrSystemGetInfo systemInfo { XR_TYPE_SYSTEM_GET_INFO };
	systemInfo.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
	if (!XRCHECK(xrGetSystem(instance, &systemInfo, &systemId)))
		return false;
	PFN_xrGetOpenGLESGraphicsRequirementsKHR getRequirements = nullptr;
	if (!XRCHECK(xrGetInstanceProcAddr(instance, "xrGetOpenGLESGraphicsRequirementsKHR", (PFN_xrVoidFunction *)&getRequirements)))
		return false;
	XrGraphicsRequirementsOpenGLESKHR requirements { XR_TYPE_GRAPHICS_REQUIREMENTS_OPENGL_ES_KHR };
	if (!XRCHECK(getRequirements(instance, systemId, &requirements)))
		return false;

	// Bind the session to Flycast's own GL context, current on this thread.
	EGLDisplay display = eglGetCurrentDisplay();
	EGLContext context = eglGetCurrentContext();
	EGLint configId = 0, configCount = 0;
	EGLConfig eglConfig = nullptr;
	if (context == EGL_NO_CONTEXT || !eglQueryContext(display, context, EGL_CONFIG_ID, &configId))
	{
		ERROR_LOG(RENDERER, "XR: no current EGL context");
		return false;
	}
	const EGLint configAttribs[] { EGL_CONFIG_ID, configId, EGL_NONE };
	if (!eglChooseConfig(display, configAttribs, &eglConfig, 1, &configCount) || configCount == 0)
	{
		ERROR_LOG(RENDERER, "XR: EGL config not found");
		return false;
	}
	XrGraphicsBindingOpenGLESAndroidKHR binding { XR_TYPE_GRAPHICS_BINDING_OPENGL_ES_ANDROID_KHR };
	binding.display = display;
	binding.config = eglConfig;
	binding.context = context;
	XrSessionCreateInfo sessionInfo { XR_TYPE_SESSION_CREATE_INFO };
	sessionInfo.next = &binding;
	sessionInfo.systemId = systemId;
	if (!XRCHECK(xrCreateSession(instance, &sessionInfo, &session)))
		return false;

	XrReferenceSpaceCreateInfo spaceInfo { XR_TYPE_REFERENCE_SPACE_CREATE_INFO };
	spaceInfo.referenceSpaceType = XR_REFERENCE_SPACE_TYPE_LOCAL;
	spaceInfo.poseInReferenceSpace.orientation.w = 1.f;
	if (!XRCHECK(xrCreateReferenceSpace(session, &spaceInfo, &localSpace)))
		return false;
	if (!initActions() || !initSwapchains())
		return false;
	NOTICE_LOG(RENDERER, "XR: session created");
	return true;
}

void pollEvents()
{
	XrEventDataBuffer event { XR_TYPE_EVENT_DATA_BUFFER };
	while (xrPollEvent(instance, &event) == XR_SUCCESS)
	{
		if (event.type == XR_TYPE_EVENT_DATA_SESSION_STATE_CHANGED)
		{
			const auto& change = *reinterpret_cast<XrEventDataSessionStateChanged *>(&event);
			NOTICE_LOG(RENDERER, "XR: session state %d", change.state);
			if (change.state == XR_SESSION_STATE_READY)
			{
				XrSessionBeginInfo begin { XR_TYPE_SESSION_BEGIN_INFO };
				begin.primaryViewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
				if (XRCHECK(xrBeginSession(session, &begin)))
				{
					sessionRunning = true;
					if (requestRefreshRate)
						XRCHECK(requestRefreshRate(session, (float)config::VrXrRefreshRate));
					if (setPerformanceLevel)
					{
						XRCHECK(setPerformanceLevel(session, XR_PERF_SETTINGS_DOMAIN_CPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT));
						XRCHECK(setPerformanceLevel(session, XR_PERF_SETTINGS_DOMAIN_GPU_EXT, XR_PERF_SETTINGS_LEVEL_SUSTAINED_HIGH_EXT));
					}
				}
			}
			else if (change.state == XR_SESSION_STATE_STOPPING)
			{
				xrEndSession(session);
				sessionRunning = false;
			}
			else if (change.state == XR_SESSION_STATE_LOSS_PENDING)
				sessionLost = true;
			else if (change.state == XR_SESSION_STATE_EXITING)
				exiting = true;
			const bool nowFocused = change.state == XR_SESSION_STATE_FOCUSED;
			if (nowFocused && !focused && originSet
					&& std::chrono::steady_clock::now() - focusLost > std::chrono::seconds(3))
				// back after a while (headset put on again, long menu visit): the player
				// may face elsewhere now
				recenterRequested = true;
			if (focused && !nowFocused)
				focusLost = std::chrono::steady_clock::now();
			focused = nowFocused;
		}
		else if (event.type == XR_TYPE_EVENT_DATA_INSTANCE_LOSS_PENDING)
		{
			sessionLost = true;
		}
		else if (event.type == XR_TYPE_EVENT_DATA_REFERENCE_SPACE_CHANGE_PENDING)
		{
			recenterRequested = true;
		}
		event = { XR_TYPE_EVENT_DATA_BUFFER };
	}
}

glm::vec3 toVec(const XrVector3f& v) { return { v.x, v.y, v.z }; }
glm::quat toQuat(const XrQuaternionf& q) { return glm::quat(q.w, q.x, q.y, q.z); }

glm::quat yawOnly(const glm::quat& q)
{
	const glm::vec3 forward = q * glm::vec3(0, 0, -1);
	return glm::angleAxis(std::atan2(-forward.x, -forward.z), glm::vec3(0, 1, 0));
}

// Pose relative to the game camera's place in the room.
glm::vec3 relativePosition(const XrVector3f& p) {
	return glm::conjugate(originRot) * (toVec(p) - originPos);
}
glm::quat relativeRotation(const XrQuaternionf& q) {
	return glm::conjugate(originRot) * toQuat(q);
}

glm::mat4 eyeProjection(const XrFovf& fov)
{
	const float l = std::tan(fov.angleLeft), r = std::tan(fov.angleRight);
	const float u = std::tan(fov.angleUp), d = std::tan(fov.angleDown);
	glm::mat4 p(0.f);
	p[0][0] = 2.f / (r - l);
	p[1][1] = 2.f / (u - d);
	p[2][0] = (r + l) / (r - l);
	p[2][1] = (u + d) / (u - d);
	p[2][2] = -1.f;
	p[2][3] = -1.f;
	p[3][2] = -2.f * NearMetres;	// infinite far plane; depth comes from the game's own W
	return p;
}

// Rebuilt game eye space -> headset local space, relative to the origin.
glm::mat4 gameToWorld()
{
	const float s = config::VrWorldScale;
	return glm::scale(glm::mat4(1.f), glm::vec3(s, -s, s));
}

void updateOrigin(bool tracked)
{
	if (!tracked || (originSet && !recenterRequested))
		return;
	const glm::vec3 center = (toVec(views[0].pose.position) + toVec(views[1].pose.position)) * 0.5f;
	originPos = center;
	originRot = yawOnly(toQuat(views[0].pose.orientation));
	originSet = true;
	recenterRequested = false;
	NOTICE_LOG(RENDERER, "XR: game camera placed at the head");
}

// Möller-Trumbore; returns the distance along the ray, or a negative value for a miss.
float rayTriangle(const glm::vec3& o, const glm::vec3& d, const glm::vec3& a, const glm::vec3& b, const glm::vec3& c)
{
	const glm::vec3 e1 = b - a, e2 = c - a;
	const glm::vec3 p = glm::cross(d, e2);
	const float det = glm::dot(e1, p);
	if (std::abs(det) < 1e-9f)
		return -1.f;
	const float inv = 1.f / det;
	const glm::vec3 t = o - a;
	const float u = glm::dot(t, p) * inv;
	if (u < 0.f || u > 1.f)
		return -1.f;
	const glm::vec3 q = glm::cross(t, e1);
	const float v = glm::dot(d, q) * inv;
	if (v < 0.f || u + v > 1.f)
		return -1.f;
	return glm::dot(e2, q) * inv;
}

// Closer than this (game units) is the game's clamped near plane, not real geometry.
constexpr float MinSceneW = 0.5f;

// How the last aim was placed, for the shot log.
const char *aimHow = "";

bool onGameScreen(const glm::vec2& screen) {
	return screen.x >= 0.f && screen.x <= 1.f && screen.y >= 0.f && screen.y <= 1.f;
}

// Where the ray (game eye space) meets the plane the 2D overlay and flat screens are
// shown on, at their stock framing (see the VR_REPROJECT vertex shader and drawVrScreen):
// the position in screen ndc (-1..1, y down) and the distance along the ray.
bool hitOverlayPlane(const glm::vec3& o, const glm::vec3& d, const glm::vec2& dcSize, glm::vec2& ndc, float& dist)
{
	const float depth = config::VrHudDepth;
	if (d.z > -1e-3f)
		return false;
	dist = (-depth - o.z) / d.z;
	if (dist <= 0.f)
		return false;
	const glm::vec3 hit = o + d * dist;
	const glm::vec2 halfSize = dcSize * 0.5f / (float)config::VrFocal * depth;
	ndc = glm::vec2(hit.x, hit.y) / halfSize;
	return true;
}

bool inTriangle(const glm::vec2& p, const glm::vec2& a, const glm::vec2& b, const glm::vec2& c)
{
	auto side = [](const glm::vec2& p, const glm::vec2& a, const glm::vec2& b) {
		return (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
	};
	const float s1 = side(p, a, b), s2 = side(p, b, c), s3 = side(p, c, a);
	return (s1 >= 0.f && s2 >= 0.f && s3 >= 0.f) || (s1 <= 0.f && s2 <= 0.f && s3 <= 0.f);
}

// Where on the game screen (0..1) the gun points, as the game's own hit test will see it,
// and how far along the ray the aim point is (game units).
//  - 2D overlay (menu items, text) is on its own plane at the stock framing: when the ray
//    passes through a piece of it, the shot goes there. So do all shots on a screen with
//    nothing but overlay (menus).
//  - Otherwise the nearest 3D surface, placed on the screen the game sees: its live
//    (possibly widened) view.
//  - Nothing hit: the point far along the ray.
bool aimAtScene(const rend_context& ctx, const glm::vec3& o, const glm::vec3& d, glm::vec2& screen, float& dist)
{
	const GameCamera cam = gameCamera(ctx);
	const glm::vec3 comfort = comfortParams();
	static std::vector<glm::vec3> pos;
	static std::vector<glm::vec2> overlayNdc;
	enum Kind : u8 { Unused, Scene, Overlay };
	static std::vector<Kind> kind;
	pos.resize(ctx.verts.size());
	overlayNdc.resize(ctx.verts.size());
	kind.assign(ctx.verts.size(), Unused);
	for (size_t i = 0; i < ctx.verts.size(); i++)
	{
		const Vertex& v = ctx.verts[i];
		if (!(v.z > 0.f) || !std::isfinite(v.z))
			continue;
		const float w = 1.f / v.z;
		const glm::vec2 ndc(v.x * 2.f / cam.dcSize.x - 1.f, v.y * 2.f / cam.dcSize.y - 1.f);
		if (std::abs(w - cam.overlayW) < 0.002f)
		{
			// hidden letterbox bars aren't there to shoot at
			if (config::VrHideLetterbox && v.col[3] > 250 && v.col[0] < 3 && v.col[1] < 3 && v.col[2] < 3)
				continue;
			overlayNdc[i] = ndc;
			kind[i] = Overlay;
		}
		else if (w >= MinSceneW)
		{
			// where it is seen: the comfort zone pulls close things back (along the line of
			// sight from the game camera, so the screen position stays the same)
			const glm::vec3 p(ndc.x * w * cam.tanHalf.x, ndc.y * w * cam.tanHalf.y, -w);
			pos[i] = p * comfortScale(p, comfort);
			kind[i] = Scene;
		}
		// else: a vertex the game clamped to its near plane (W 0.001, huge x/y). Its
		// triangles turn into long invisible slivers across the view that would catch the
		// aim; the visible part of such a polygon is hit through its other triangles.
	}

	glm::vec2 planeNdc;
	float planeDist = 0.f;
	const bool onPlane = hitOverlayPlane(o, d, cam.dcSize, planeNdc, planeDist);
	bool overlayHit = false;
	bool hasScene = false;
	float nearest = 1e30f;
	auto triangle = [&](u32 a, u32 b, u32 c, bool solid, bool background) {
		if (a == RestartIndex || b == RestartIndex || c == RestartIndex
				|| a >= kind.size() || b >= kind.size() || c >= kind.size())
			return;
		if (kind[a] == Overlay && kind[b] == Overlay && kind[c] == Overlay)
		{
			// skip full-screen fades and backdrops: only buttons and text count
			const glm::vec2 e1 = overlayNdc[b] - overlayNdc[a], e2 = overlayNdc[c] - overlayNdc[a];
			if (onPlane && !overlayHit && std::abs(e1.x * e2.y - e1.y * e2.x) < 1.f
					&& inTriangle(planeNdc, overlayNdc[a], overlayNdc[b], overlayNdc[c]))
				overlayHit = true;
		}
		else if (solid && kind[a] == Scene && kind[b] == Scene && kind[c] == Scene)
		{
			if (!background)
				hasScene = true;
			const float t = rayTriangle(o, d, pos[a], pos[b], pos[c]);
			if (t > 0.f && t < nearest)
				nearest = t;
		}
	};
	// strips: first/count are ranges of the index buffer
	auto strips = [&](const std::vector<PolyParam>& list, size_t from, size_t to, bool solid, bool opaque) {
		for (size_t n = from; n < to && n < list.size(); n++)
		{
			const PolyParam& pp = list[n];
			for (u32 j = pp.first; j + 2 < pp.first + pp.count && j + 2 < ctx.idx.size(); j++)
				// the first opaque polygon is the background plane
				triangle(ctx.idx[j], ctx.idx[j + 1], ctx.idx[j + 2], solid, opaque && n == 0);
		}
	};
	strips(ctx.global_param_op, 0, ctx.global_param_op.size(), true, true);
	strips(ctx.global_param_pt, 0, ctx.global_param_pt.size(), true, false);
	// Translucent polygons (most 2D text) of auto-sorted passes were turned into a list of
	// sorted triangles; their own first/count still point at vertices, not indices.
	RenderPass prev {};
	for (const RenderPass& pass : ctx.render_passes)
	{
		if (pass.sorted_tr_count != prev.sorted_tr_count)
		{
			for (u32 s = prev.sorted_tr_count; s < pass.sorted_tr_count && s < ctx.sortedTriangles.size(); s++)
			{
				const SortedTriangle& st = ctx.sortedTriangles[s];
				for (u32 j = st.first; j + 2 < st.first + st.count && j + 2 < ctx.idx.size(); j += 3)
					triangle(ctx.idx[j], ctx.idx[j + 1], ctx.idx[j + 2], false, false);
			}
		}
		else
			strips(ctx.global_param_tr, prev.tr_count, pass.tr_count, false, false);
		prev = pass;
	}

	if (onPlane && (overlayHit || !hasScene))
	{
		screen = planeNdc * 0.5f + 0.5f;
		dist = planeDist;
		aimHow = overlayHit ? "2D" : "menu";
		return onGameScreen(screen);
	}
	aimHow = nearest < 1e30f ? "3D" : "far";
	constexpr float Far = 10000.f;
	dist = nearest < Far ? nearest : Far;
	const glm::vec3 hit = o + d * dist;
	if (hit.z > -1e-3f)
		return false;
	// The game tests gun hits with its stock view, not the widened one it draws with
	// (HOTD2: shots drifted towards the centre, more so near the edges, right where its
	// own shot flash showed up). So the hit point goes on the screen at the stock framing:
	// the shot lands where the dot is, and outside the stock view is off screen, as in
	// the original game.
	const glm::vec2 tan = config::VrAimWidened ? cam.tanHalf : cam.dcSize * 0.5f / (float)config::VrFocal;
	screen = glm::vec2(hit.x / (-hit.z * tan.x), hit.y / (-hit.z * tan.y)) * 0.5f + 0.5f;
	return onGameScreen(screen);
}

// The flat screen menus and notices are shown on (see drawVrScreen in gles.cpp).
bool aimAtFlatScreen(const glm::vec3& o, const glm::vec3& d, glm::vec2& screen, float& dist)
{
	glm::vec2 ndc;
	if (!hitOverlayPlane(o, d, glm::vec2(640.f, 480.f), ndc, dist))
		return false;
	screen = ndc * 0.5f + 0.5f;
	aimHow = "screen";
	return onGameScreen(screen);
}

bool boolAction(XrAction action)
{
	XrActionStateGetInfo info { XR_TYPE_ACTION_STATE_GET_INFO };
	info.action = action;
	XrActionStateBoolean state { XR_TYPE_ACTION_STATE_BOOLEAN };
	return XR_SUCCEEDED(xrGetActionStateBoolean(session, &info, &state)) && state.isActive && state.currentState;
}

float floatAction(XrAction action, int hand)
{
	XrActionStateGetInfo info { XR_TYPE_ACTION_STATE_GET_INFO };
	info.action = action;
	info.subactionPath = handPaths[hand];
	XrActionStateFloat state { XR_TYPE_ACTION_STATE_FLOAT };
	return XR_SUCCEEDED(xrGetActionStateFloat(session, &info, &state)) && state.isActive ? state.currentState : 0.f;
}

void recoil()
{
	XrHapticActionInfo info { XR_TYPE_HAPTIC_ACTION_INFO };
	info.action = hapticAction;
	info.subactionPath = handPaths[gunHand];
	XrHapticVibration vibration { XR_TYPE_HAPTIC_VIBRATION };
	vibration.duration = 40'000'000;	// 40 ms
	vibration.frequency = XR_FREQUENCY_UNSPECIFIED;
	vibration.amplitude = 0.8f;
	xrApplyHapticFeedback(session, &info, (const XrHapticBaseHeader *)&vibration);
}

glm::vec2 stickAction2()
{
	XrActionStateGetInfo info { XR_TYPE_ACTION_STATE_GET_INFO };
	info.action = stickAction;
	XrActionStateVector2f stick { XR_TYPE_ACTION_STATE_VECTOR2F };
	if (!XR_SUCCEEDED(xrGetActionStateVector2f(session, &info, &stick)) || !stick.isActive)
		return glm::vec2(0.f);
	return glm::vec2(stick.currentState.x, stick.currentState.y);
}

// The light gun's D-pad from a thumbstick, with some hysteresis: pushed past 60%,
// let go below 40%. Returns the lightgunSet bits (up 8, down 16, left 32, right 64).
u32 dpadFromStick(const glm::vec2& stick)
{
	static u32 held;
	auto axis = [&](float v, u32 negBit, u32 posBit) {
		const u32 was = held & (negBit | posBit);
		if (was == posBit ? v > 0.4f : v > 0.6f)
			return posBit;
		if (was == negBit ? v < -0.4f : v < -0.6f)
			return negBit;
		return 0u;
	};
	held = axis(stick.y, 16, 8) | axis(stick.x, 32, 64);
	return held;
}

// With a grip held, thumbstick up makes the world bigger around the player, down smaller
// (about 1.8x per second held). The size is kept in emu.cfg when the stick is let go.
// Returns true while sizing (the stick then isn't the D-pad).
bool adjustWorldSize(XrTime time, const glm::vec2& stick)
{
	static XrTime lastTime;
	static bool changed;
	const float dt = lastTime != 0 ? std::clamp((time - lastTime) * 1e-9f, 0.f, 0.1f) : 0.f;
	lastTime = time;
	XrActionStateGetInfo info { XR_TYPE_ACTION_STATE_GET_INFO };
	info.action = gripAction;
	XrActionStateFloat grip { XR_TYPE_ACTION_STATE_FLOAT };
	const bool gripped = XR_SUCCEEDED(xrGetActionStateFloat(session, &info, &grip)) && grip.isActive && grip.currentState > 0.7f;
	const float y = gripped ? stick.y : 0.f;
	if (std::abs(y) > 0.5f)
	{
		const float scale = std::clamp(config::VrWorldScale * std::exp((y > 0.f ? 0.6f : -0.6f) * dt), 0.005f, 0.2f);
		config::VrWorldScale.set(scale);
		changed = true;
	}
	else if (changed)
	{
		changed = false;
		config::saveFloat("config", "vr.WorldScale", config::VrWorldScale);
		NOTICE_LOG(RENDERER, "XR: world size %.4f m per game unit", (float)config::VrWorldScale);
	}
	return gripped;
}

// Off the game screen, as a light gun sees it (any negative position).
constexpr int OffScreen = -10000;

void updateLightgun(XrTime time)
{
	gunVisible = false;
	if (!config::VrXrGun)
		return;
	if (!focused)
	{
		// headset menu or app in the background: let go of everything
		lightgunSet(0, OffScreen, OffScreen, 0);
		triggerWas = false;
		return;
	}
	const XrActiveActionSet active { actionSet, XR_NULL_PATH };
	XrActionsSyncInfo sync { XR_TYPE_ACTIONS_SYNC_INFO };
	sync.countActiveActionSets = 1;
	sync.activeActionSets = &active;
	if (!XR_SUCCEEDED(xrSyncActions(session, &sync)))
		return;

	const bool recenter = boolAction(recenterAction);
	if (recenter && !recenterWas)
		recenterRequested = true;
	recenterWas = recenter;
	const glm::vec2 stick = stickAction2();
	const u32 dpad = adjustWorldSize(time, stick) ? dpadFromStick(glm::vec2(0.f)) : dpadFromStick(stick);

	// Which hand holds the gun: the one whose trigger is pulled while the gun hand's isn't,
	// or the other one when the gun hand's controller is gone (asleep, battery).
	const XrSpaceLocationFlags valid = XR_SPACE_LOCATION_POSITION_VALID_BIT | XR_SPACE_LOCATION_ORIENTATION_VALID_BIT;
	XrSpaceLocation locations[2] { { XR_TYPE_SPACE_LOCATION }, { XR_TYPE_SPACE_LOCATION } };
	bool tracked[2];
	float pulls[2];
	for (int hand : { Left, Right })
	{
		tracked[hand] = XR_SUCCEEDED(xrLocateSpace(aimSpaces[hand], localSpace, time, &locations[hand]))
				&& (locations[hand].locationFlags & valid) == valid;
		pulls[hand] = tracked[hand] ? floatAction(triggerAction, hand) : 0.f;
	}
	const int otherHand = 1 - gunHand;
	if (!triggerWas && ((tracked[otherHand] && !tracked[gunHand]) || (pulls[otherHand] > 0.6f && pulls[gunHand] < 0.35f)))
	{
		gunHand = otherHand;
		NOTICE_LOG(INPUT, "XR: the gun is in the %s hand", gunHand == Right ? "right" : "left");
	}

	// A clean click: pulled past 60%, let go below 35%, so a resting finger or a half
	// release doesn't fire twice.
	const float pull = pulls[gunHand];
	const bool trigger = triggerWas ? pull > 0.35f : pull > 0.6f;

	glm::vec2 screen(-1.f);
	bool onScreen = false;
	const XrSpaceLocation& location = locations[gunHand];
	if (originSet && tracked[gunHand])
	{
		const glm::quat rot = relativeRotation(location.pose.orientation);
		const glm::mat4 pose = glm::translate(glm::mat4(1.f), relativePosition(location.pose.position)) * glm::mat4_cast(rot);
		// Shots leave the muzzle along the barrel, so the aim line and the hit agree. Into
		// rebuilt game eye space for the hit test.
		const glm::vec3 o = glm::vec3(pose * glm::vec4(GunMuzzle, 1.f));
		const glm::vec3 d = rot * glm::vec3(0, 0, -1);
		const float s = config::VrWorldScale;
		const glm::vec3 og = glm::vec3(o.x, -o.y, o.z) / s;
		const glm::vec3 dg = glm::normalize(glm::vec3(d.x, -d.y, d.z));
		float dist = -1.f;
		const rend_context *ctx = gles_vr_frame();
		if (gles_vr_showing_framebuffer())
			onScreen = aimAtFlatScreen(og, dg, screen, dist);
		else if (ctx != nullptr)
			onScreen = aimAtScene(*ctx, og, dg, screen, dist);

		// Recoil: back and muzzle up around the hand, settling in about a tenth of a second.
		// A shot fired now shows its flash and kick in this very frame.
		if (trigger && !triggerWas && onScreen)
			lastShot = time;
		const float sinceShot = lastShot != 0 ? (time - lastShot) * 1e-9f : 1.f;
		const float kick = sinceShot >= 0.f && sinceShot < 0.4f ? std::exp(-sinceShot / 0.055f) : 0.f;
		const glm::vec3 hand(0.f, -0.06f, 0.06f);
		glm::mat4 recoilMat = glm::translate(glm::mat4(1.f), glm::vec3(0.f, 0.f, 0.03f * kick) + hand);
		recoilMat = glm::rotate(recoilMat, glm::radians(16.f * kick), glm::vec3(1, 0, 0));
		recoilMat = glm::translate(recoilMat, -hand);
		gunView.pose = pose * recoilMat;
		gunView.trigger = std::clamp(pull, 0.f, 1.f);
		gunView.flash = sinceShot >= 0.f && sinceShot < 0.06f ? 1.f - sinceShot / 0.06f : 0.f;
		gunView.aimLine = config::VrLaser;
		gunView.aimDot = config::VrLaser && onScreen;
		gunView.aimPoint = o + d * (dist > 0.f ? dist * s : 5.f);
		gunVisible = config::VrShowGun;
	}

	u32 buttons = 0;
	if (trigger)
		// pointing away from the screen fires off-screen: that's how you reload
		buttons |= onScreen ? 1 : 2;
	if (boolAction(reloadAction))
		buttons |= 2;
	if (boolAction(startAction))
		buttons |= 4;
	buttons |= dpad;
	if (trigger && !triggerWas)
	{
		recoil();
		// diagnostics: where shots go, to compare with what the game makes of them
		if (onScreen)
			NOTICE_LOG(INPUT, "XR: shot at %d,%d (%s, %d cm)", (int)(screen.x * 640.f), (int)(screen.y * 480.f), aimHow,
					(int)(glm::length(gunView.aimPoint - glm::vec3(gunView.pose[3])) * 100.f));
		else
			NOTICE_LOG(INPUT, "XR: shot off screen (reload)");
	}
	triggerWas = trigger;
	const glm::ivec2 pos = onScreen ? glm::ivec2(screen * 10000.f) : glm::ivec2(OffScreen);
	lightgunSet(0, pos.x, pos.y, buttons);
}

// Debug aid: "touch files/capture.request" (run-as) and the next left-eye image is
// written to files/eye0.ppm.
void captureIfRequested(GLuint fbo, int width, int height)
{
	static std::string dir;
	if (dir.empty())
	{
		// the app's internal files directory, from the game path we were started with
		const std::string& game = settings.content.path;
		const size_t at = game.find("/files/");
		dir = at == std::string::npos ? "/data/local/tmp" : game.substr(0, at + 6);
	}
	const std::string request = dir + "/capture.request";
	if (access(request.c_str(), F_OK) != 0)
		return;
	unlink(request.c_str());
	std::vector<u8> pixels((size_t)width * height * 4);
	glBindFramebuffer(GL_FRAMEBUFFER, fbo);
	glPixelStorei(GL_PACK_ALIGNMENT, 1);
	glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
	FILE *f = fopen((dir + "/eye0.ppm").c_str(), "wb");
	if (f == nullptr)
		return;
	fprintf(f, "P6\n%d %d\n255\n", width, height);
	for (int y = height - 1; y >= 0; y--)
		for (int x = 0; x < width; x++)
			fwrite(&pixels[((size_t)y * width + x) * 4], 1, 3, f);
	fclose(f);
	NOTICE_LOG(RENDERER, "XR: captured the left eye to %s/eye0.ppm", dir.c_str());

	// ...and the picture the console's video output reads from VRAM right now, which is
	// what a TV would show when the game draws with the CPU.
	FramebufferInfo info;
	info.update();
	PixelBuffer<u32> pb;
	int fbWidth = 0, fbHeight = 0;
	ReadFramebuffer<RGBAPacker>(info, pb, fbWidth, fbHeight);
	f = fbWidth > 0 && fbHeight > 0 ? fopen((dir + "/fb0.ppm").c_str(), "wb") : nullptr;
	if (f == nullptr)
		return;
	fprintf(f, "P6\n%d %d\n255\n", fbWidth, fbHeight);
	const u32 *px = pb.data();
	for (int i = 0; i < fbWidth * fbHeight; i++)
		fwrite(&px[i], 1, 3, f);
	fclose(f);
	NOTICE_LOG(RENDERER, "XR: video output %dx%d from %06x to %s/fb0.ppm", fbWidth, fbHeight, info.fb_r_sof1, dir.c_str());

	// ...and the shown frame's solid 3D triangles, rebuilt in game eye space (y down, game
	// units, integers in 1/1000), for measuring the scene (floor height, sizes) offline.
	const rend_context *ctx = gles_vr_frame();
	FILE *s = ctx != nullptr ? fopen((dir + "/scene.txt").c_str(), "w") : nullptr;
	if (s == nullptr)
		return;
	const GameCamera cam = gameCamera(*ctx);
	fprintf(s, "tan %d %d scale %d\n", (int)(cam.tanHalf.x * 1e6f), (int)(cam.tanHalf.y * 1e6f), (int)(config::VrWorldScale * 1e6f));
	auto vertex = [&](u32 i, glm::vec3& p) {
		if (i == RestartIndex || i >= ctx->verts.size())
			return false;
		const Vertex& v = ctx->verts[i];
		if (!(v.z > 0.f) || !std::isfinite(v.z))
			return false;
		const float w = 1.f / v.z;
		if (w < MinSceneW || std::abs(w - cam.overlayW) < 0.002f)
			return false;
		const glm::vec2 ndc(v.x * 2.f / cam.dcSize.x - 1.f, v.y * 2.f / cam.dcSize.y - 1.f);
		p = glm::vec3(ndc.x * w * cam.tanHalf.x, ndc.y * w * cam.tanHalf.y, -w);
		return true;
	};
	int count = 0;
	for (const std::vector<PolyParam> *list : { &ctx->global_param_op, &ctx->global_param_pt })
		for (size_t n = list == &ctx->global_param_op ? 1 : 0; n < list->size(); n++)
		{
			const PolyParam& pp = (*list)[n];
			for (u32 j = pp.first; j + 2 < pp.first + pp.count && j + 2 < ctx->idx.size(); j++)
			{
				glm::vec3 a, b, c;
				if (!vertex(ctx->idx[j], a) || !vertex(ctx->idx[j + 1], b) || !vertex(ctx->idx[j + 2], c))
					continue;
				fprintf(s, "%d %d %d %d %d %d %d %d %d\n",
						(int)(a.x * 1000), (int)(a.y * 1000), (int)(a.z * 1000),
						(int)(b.x * 1000), (int)(b.y * 1000), (int)(b.z * 1000),
						(int)(c.x * 1000), (int)(c.y * 1000), (int)(c.z * 1000));
				count++;
			}
		}
	fclose(s);
	NOTICE_LOG(RENDERER, "XR: %d scene triangles to %s/scene.txt", count, dir.c_str());
}

}	// namespace

bool enabled()
{
	static const bool on = config::VrXr;
	return on && !(initTried && session == XR_NULL_HANDLE);
}

const Eye *currentEye()
{
	return drawingEye ? &eye : nullptr;
}

void term()
{
	if (instance != XR_NULL_HANDLE)
	{
		NOTICE_LOG(RENDERER, "XR: render context going away, closing the session");
		// still on the render thread with the context current: our own GL objects first
		termGun();
		for (EyeChain& c : chains)
		{
			if (!c.fbos.empty())
				glDeleteFramebuffers((GLsizei)c.fbos.size(), c.fbos.data());
			if (c.depthStencil != 0)
				glDeleteRenderbuffers(1, &c.depthStencil);
		}
		if (session != XR_NULL_HANDLE)
			xrDestroySession(session);	// with its spaces and swapchains
		xrDestroyInstance(instance);	// with the actions
		// let go of the gun
		if (config::VrXrGun)
			lightgunSet(0, OffScreen, OffScreen, 0);
	}
	for (EyeChain& c : chains)
		c = EyeChain();
	instance = XR_NULL_HANDLE;
	systemId = XR_NULL_SYSTEM_ID;
	session = XR_NULL_HANDLE;
	localSpace = aimSpaces[Left] = aimSpaces[Right] = XR_NULL_HANDLE;
	actionSet = XR_NULL_HANDLE;
	aimAction = triggerAction = startAction = reloadAction = recenterAction = hapticAction = stickAction = gripAction = XR_NULL_HANDLE;
	requestRefreshRate = nullptr;
	setPerformanceLevel = nullptr;
	sessionRunning = focused = drawingEye = false;
	sessionLost = exiting = false;
	originSet = recenterRequested = false;
	triggerWas = recenterWas = false;
	gunVisible = false;
	lastShot = 0;
	initTried = false;
}

bool frame()
{
	if (!initTried && !init())
	{
		ERROR_LOG(RENDERER, "XR: initialisation failed, showing the flat window instead");
		return rend_single_frame(true);
	}
	if (session != XR_NULL_HANDLE)
		pollEvents();
	if (sessionLost || exiting)
	{
		// Lost: a new session next frame. Exiting (quit from the headset menu): none.
		const bool quit = exiting;
		NOTICE_LOG(RENDERER, "XR: session %s", quit ? "exiting" : "lost");
		term();
		initTried = quit;
		exiting = quit;
	}
	if (!sessionRunning)
	{
		// Not shown (yet): keep the emulator's frames moving without spinning.
		rend_vr_drain(10);
		return false;
	}

	XrFrameWaitInfo waitInfo { XR_TYPE_FRAME_WAIT_INFO };
	XrFrameState frameState { XR_TYPE_FRAME_STATE };
	XrResult result = xrWaitFrame(session, &waitInfo, &frameState);
	if (XR_SUCCEEDED(result))
	{
		XrFrameBeginInfo beginInfo { XR_TYPE_FRAME_BEGIN_INFO };
		result = xrBeginFrame(session, &beginInfo);
	}
	if (!check(result, "xrWaitFrame/xrBeginFrame"))
	{
		sessionLost = result == XR_ERROR_SESSION_LOST || result == XR_ERROR_INSTANCE_LOST;
		rend_vr_drain(10);
		return false;
	}

	// Let through whatever the emulator produced since the last headset frame.
	rend_vr_drain(0);

	XrViewState viewState { XR_TYPE_VIEW_STATE };
	XrViewLocateInfo locateInfo { XR_TYPE_VIEW_LOCATE_INFO };
	locateInfo.viewConfigurationType = XR_VIEW_CONFIGURATION_TYPE_PRIMARY_STEREO;
	locateInfo.displayTime = frameState.predictedDisplayTime;
	locateInfo.space = localSpace;
	u32 viewCount = 0;
	const XrViewStateFlags tracked = XR_VIEW_STATE_ORIENTATION_VALID_BIT | XR_VIEW_STATE_POSITION_VALID_BIT;
	const bool located = XR_SUCCEEDED(xrLocateViews(session, &locateInfo, &viewState, 2, &viewCount, views))
			&& viewCount == 2 && (viewState.viewStateFlags & tracked) == tracked;
	updateOrigin(located);
	updateLightgun(frameState.predictedDisplayTime);

	XrCompositionLayerProjectionView projViews[2] { { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW }, { XR_TYPE_COMPOSITION_LAYER_PROJECTION_VIEW } };
	const bool render = frameState.shouldRender && located;
	bool ok = true;
	for (int i = 0; render && i < 2; i++)
	{
		EyeChain& c = chains[i];
		u32 index = 0;
		XrSwapchainImageAcquireInfo acquire { XR_TYPE_SWAPCHAIN_IMAGE_ACQUIRE_INFO };
		XrSwapchainImageWaitInfo wait { XR_TYPE_SWAPCHAIN_IMAGE_WAIT_INFO };
		wait.timeout = XR_INFINITE_DURATION;
		if (!XRCHECK(xrAcquireSwapchainImage(c.chain, &acquire, &index)) || !XRCHECK(xrWaitSwapchainImage(c.chain, &wait)))
		{
			ok = false;
			break;
		}
		const glm::mat4 view = glm::inverse(glm::translate(glm::mat4(1.f), relativePosition(views[i].pose.position))
				* glm::mat4_cast(relativeRotation(views[i].pose.orientation)));
		eye = { c.fbos[index], c.width, c.height, eyeProjection(views[i].fov) * view * gameToWorld() };
		if (gles_vr_have_frame())
		{
			drawingEye = true;
			gles_vr_draw_eye(c.width, c.height);
			drawingEye = false;
		}
		else
		{
			// nothing to show yet: black, not whatever the image held before
			glBindFramebuffer(GL_FRAMEBUFFER, c.fbos[index]);
			glViewport(0, 0, c.width, c.height);
			glcache.Disable(GL_SCISSOR_TEST);
			glcache.ClearColor(0.f, 0.f, 0.f, 1.f);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		if (gunVisible)
		{
			glBindFramebuffer(GL_FRAMEBUFFER, c.fbos[index]);
			glViewport(0, 0, c.width, c.height);
			drawGun(eyeProjection(views[i].fov) * view, relativePosition(views[i].pose.position), gunView);
		}
		if (i == 0)
			captureIfRequested(c.fbos[index], c.width, c.height);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		XrSwapchainImageReleaseInfo release { XR_TYPE_SWAPCHAIN_IMAGE_RELEASE_INFO };
		if (!XRCHECK(xrReleaseSwapchainImage(c.chain, &release)))
		{
			ok = false;
			break;
		}
		projViews[i].pose = views[i].pose;
		projViews[i].fov = views[i].fov;
		projViews[i].subImage.swapchain = c.chain;
		projViews[i].subImage.imageRect.extent = { c.width, c.height };
	}

	XrCompositionLayerProjection layer { XR_TYPE_COMPOSITION_LAYER_PROJECTION };
	layer.space = localSpace;
	layer.viewCount = 2;
	layer.views = projViews;
	const XrCompositionLayerBaseHeader *layers[] { (const XrCompositionLayerBaseHeader *)&layer };
	XrFrameEndInfo endInfo { XR_TYPE_FRAME_END_INFO };
	endInfo.displayTime = frameState.predictedDisplayTime;
	endInfo.environmentBlendMode = XR_ENVIRONMENT_BLEND_MODE_OPAQUE;
	endInfo.layerCount = render && ok ? 1 : 0;
	endInfo.layers = layers;
	XRCHECK(xrEndFrame(session, &endInfo));
	return render && ok;
}

}	// namespace vr::xr
#endif
