/*
	The player's pistol in the headset (hotd2-vr). See xr_gun.h.

	The model is a Namco arcade light gun in glossy red plastic: "Namco Arcade Gun" by
	Martoscar (https://sketchfab.com/3d-models/namco-arcade-gun-15fbd5b9add94a34b5c21746e3dd32be),
	CC BY 4.0, baked into gun_model.h by tools/gun_model/convert_gun.py. It is lit per pixel
	by a fixed key light and a dim fill from below, with a plastic highlight and rim, so it
	reads as a solid object in the game's dark scenes.

	This file is part of Flycast and is distributed under the GNU GPL v2 or later.
*/
#include "xr_gun.h"
#include "gun_model.h"
#include "rend/gles/gles.h"
#include "rend/gles/glcache.h"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cstring>
#include <vector>

namespace vr::xr
{
namespace
{

static_assert(GunMuzzle.x == gunmodel::Muzzle[0] && GunMuzzle.y == gunmodel::Muzzle[1] && GunMuzzle.z == gunmodel::Muzzle[2],
		"GunMuzzle (xr_gun.h) must match the baked model");

struct Material
{
	glm::vec3 color;
	float specular;		// highlight strength
	float shininess;	// highlight tightness
	glm::vec3 emissive;
};
// per model part, by name
struct PartMaterial { const char *name; Material material; };
const PartMaterial Materials[] = {
	{ "hardsurfaces", { { 0.80f, 0.05f, 0.04f }, 0.70f, 60.f, { 0.f, 0.f, 0.f } } },		// candy red shell
	{ "softsurfaces", { { 0.56f, 0.03f, 0.03f }, 0.35f, 28.f, { 0.f, 0.f, 0.f } } },		// deeper red grip and panels
	{ "screws", { { 0.16f, 0.16f, 0.18f }, 0.8f, 80.f, { 0.f, 0.f, 0.f } } },			// dark metal
	{ "lens", { { 0.06f, 0.01f, 0.01f }, 1.0f, 140.f, { 0.12f, 0.f, 0.f } } },			// smoked lens, faint glow
};

// The model, on the GPU
GLuint modelVbo, modelIbo;
GLuint modelProgram;
GLint mMvp = -1, mModel = -1, mEye = -1, mColor = -1, mSpecular = -1, mShininess = -1, mEmissive = -1;

// Glowing bits (aim line, dot, muzzle flash): client-side, a few quads
struct GlowVertex
{
	glm::vec3 pos;
	glm::vec4 color;
	glm::vec2 corner;	// spots: -1..1 across, for the round falloff
};
GLuint glowProgram;
GLint gMvp = -1, gSpot = -1;

bool initModel()
{
	if (modelProgram != 0)
		return true;
	OpenGlSource vertex;
	vertex.addSource(VertexCompatShader).addSource(R"(
in highp vec3 in_pos;
in highp vec3 in_normal;
uniform highp mat4 mvp;
uniform highp mat4 model;
out highp vec3 vtx_pos;
out highp vec3 vtx_normal;
void main()
{
	vtx_pos = (model * vec4(in_pos, 1.0)).xyz;
	vtx_normal = mat3(model) * in_normal;
	gl_Position = mvp * vec4(in_pos, 1.0);
}
)");
	OpenGlSource fragment;
	fragment.addSource(PixelCompatShader).addSource(R"(
uniform highp vec3 eyePos;
uniform mediump vec3 baseColor;
uniform mediump float specular;
uniform mediump float shininess;
uniform mediump vec3 emissive;
in highp vec3 vtx_pos;
in highp vec3 vtx_normal;
void main()
{
	highp vec3 n = normalize(vtx_normal);
	highp vec3 v = normalize(eyePos - vtx_pos);
	if (dot(n, v) < 0.0)
		n = -n;		// inside faces seen through gaps
	const highp vec3 key = vec3(0.32, 0.86, 0.40);		// normalised: up, front, a bit right
	const highp vec3 fill = vec3(-0.45, -0.75, -0.48);
	mediump float diffuse = max(dot(n, key), 0.0);
	mediump float back = max(dot(n, fill), 0.0);
	highp vec3 h = normalize(key + v);
	mediump float highlight = pow(max(dot(n, h), 0.0), shininess) * specular;
	// (clamped: a dot a hair over 1 would make the base negative, and pow() NaN)
	highp float edge = 1.0 - clamp(dot(n, v), 0.0, 1.0);
	mediump float rim = edge * edge * edge;
	mediump vec3 color = baseColor * (0.30 + 0.80 * diffuse + 0.25 * back)
			+ vec3(highlight)
			+ rim * (0.18 * baseColor + 0.10 * specular)
			+ emissive;
	gl_FragColor = vec4(color, 1.0);
}
)");
	modelProgram = gl_CompileAndLink(vertex.generate().c_str(), fragment.generate().c_str());
	if (modelProgram == 0)
		return false;
	mMvp = glGetUniformLocation(modelProgram, "mvp");
	mModel = glGetUniformLocation(modelProgram, "model");
	mEye = glGetUniformLocation(modelProgram, "eyePos");
	mColor = glGetUniformLocation(modelProgram, "baseColor");
	mSpecular = glGetUniformLocation(modelProgram, "specular");
	mShininess = glGetUniformLocation(modelProgram, "shininess");
	mEmissive = glGetUniformLocation(modelProgram, "emissive");

	// interleaved position, normal
	std::vector<float> verts(gunmodel::VertexCount * 6);
	for (unsigned i = 0; i < gunmodel::VertexCount; i++)
	{
		for (int k = 0; k < 3; k++)
		{
			verts[i * 6 + k] = gunmodel::Positions[i * 3 + k] * gunmodel::PositionScale;
			verts[i * 6 + 3 + k] = gunmodel::Normals[i * 3 + k] / 127.f;
		}
	}
	GlVertexArray::unbind();
	glGenBuffers(1, &modelVbo);
	glBindBuffer(GL_ARRAY_BUFFER, modelVbo);
	glBufferData(GL_ARRAY_BUFFER, verts.size() * sizeof(float), verts.data(), GL_STATIC_DRAW);
	glGenBuffers(1, &modelIbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, modelIbo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(gunmodel::Indices), gunmodel::Indices, GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	return true;
}

bool initGlow()
{
	if (glowProgram != 0)
		return true;
	OpenGlSource vertex;
	vertex.addSource(VertexCompatShader).addSource(R"(
in highp vec3 in_pos;
in lowp vec4 in_base;
in highp vec2 in_uv;
uniform highp mat4 mvp;
out lowp vec4 vtx_color;
out highp vec2 vtx_corner;
void main()
{
	vtx_color = in_base;
	vtx_corner = in_uv;
	gl_Position = mvp * vec4(in_pos, 1.0);
}
)");
	OpenGlSource fragment;
	fragment.addSource(PixelCompatShader).addSource(R"(
uniform lowp float spot;
in lowp vec4 vtx_color;
in highp vec2 vtx_corner;
void main()
{
	lowp float a = vtx_color.a;
	if (spot > 0.5)
		a *= 1.0 - smoothstep(0.25, 1.0, length(vtx_corner));
	gl_FragColor = vec4(vtx_color.rgb, a);
}
)");
	glowProgram = gl_CompileAndLink(vertex.generate().c_str(), fragment.generate().c_str());
	if (glowProgram == 0)
		return false;
	gMvp = glGetUniformLocation(glowProgram, "mvp");
	gSpot = glGetUniformLocation(glowProgram, "spot");
	return true;
}

void drawModel(const glm::mat4& viewProj, const glm::vec3& eyePos, const glm::mat4& pose)
{
	glcache.UseProgram(modelProgram);
	const glm::mat4 mvp = viewProj * pose;
	glUniformMatrix4fv(mMvp, 1, GL_FALSE, &mvp[0][0]);
	glUniformMatrix4fv(mModel, 1, GL_FALSE, &pose[0][0]);
	glUniform3f(mEye, eyePos.x, eyePos.y, eyePos.z);
	glBindBuffer(GL_ARRAY_BUFFER, modelVbo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, modelIbo);
	glVertexAttribPointer(VERTEX_POS_ARRAY, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (const void *)0);
	glVertexAttribPointer(VERTEX_NORM_ARRAY, 3, GL_FLOAT, GL_FALSE, 6 * sizeof(float), (const void *)(3 * sizeof(float)));
	glEnableVertexAttribArray(VERTEX_POS_ARRAY);
	glEnableVertexAttribArray(VERTEX_NORM_ARRAY);
	for (const gunmodel::Part& part : gunmodel::Parts)
	{
		const Material *m = &Materials[0].material;
		for (const PartMaterial& pm : Materials)
			if (!strcmp(pm.name, part.name))
				m = &pm.material;
		glUniform3f(mColor, m->color.r, m->color.g, m->color.b);
		glUniform1f(mSpecular, m->specular);
		glUniform1f(mShininess, m->shininess);
		glUniform3f(mEmissive, m->emissive.r, m->emissive.g, m->emissive.b);
		glDrawElements(GL_TRIANGLES, (GLsizei)part.count, GL_UNSIGNED_SHORT, (const void *)(part.first * sizeof(unsigned short)));
	}
	glDisableVertexAttribArray(VERTEX_POS_ARRAY);
	glDisableVertexAttribArray(VERTEX_NORM_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

void drawGlow(const std::vector<GlowVertex>& verts)
{
	if (verts.empty())
		return;
	const GlowVertex *v = verts.data();
	glVertexAttribPointer(VERTEX_POS_ARRAY, 3, GL_FLOAT, GL_FALSE, sizeof(GlowVertex), &v->pos);
	glVertexAttribPointer(VERTEX_COL_BASE_ARRAY, 4, GL_FLOAT, GL_FALSE, sizeof(GlowVertex), &v->color);
	glVertexAttribPointer(VERTEX_UV_ARRAY, 2, GL_FLOAT, GL_FALSE, sizeof(GlowVertex), &v->corner);
	glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());
}

// A quad facing the eye, around a point in room space.
void billboard(std::vector<GlowVertex>& out, const glm::vec3& center, const glm::vec3& eyePos, float size, const glm::vec4& color)
{
	const glm::vec3 toEye = glm::normalize(eyePos - center);
	glm::vec3 right = glm::cross(glm::vec3(0, 1, 0), toEye);
	right = glm::length(right) < 1e-4f ? glm::vec3(1, 0, 0) : glm::normalize(right);
	const glm::vec3 up = glm::cross(toEye, right);
	const glm::vec3 r = right * size, u = up * size;
	const GlowVertex a { center - r - u, color, { -1, -1 } }, b { center + r - u, color, { 1, -1 } };
	const GlowVertex c { center + r + u, color, { 1, 1 } }, d { center - r + u, color, { -1, 1 } };
	for (const GlowVertex& v : { a, b, c, a, c, d })
		out.push_back(v);
}

}	// namespace

void drawGun(const glm::mat4& viewProj, const glm::vec3& eyePos, const GunView& gun)
{
	if (!initModel() || !initGlow())
		return;

	// The gun is in the player's hand, in front of anything in the game: draw it over the
	// image with its own depth test only.
	glcache.Disable(GL_SCISSOR_TEST);
	glcache.Disable(GL_STENCIL_TEST);
	glcache.Disable(GL_CULL_FACE);
	glcache.Disable(GL_BLEND);
	glcache.Enable(GL_DEPTH_TEST);
	glcache.DepthMask(GL_TRUE);
	glcache.DepthFunc(GL_LESS);
	glClearDepthf(1.f);
	glClear(GL_DEPTH_BUFFER_BIT);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	GlVertexArray::unbind();

	drawModel(viewProj, eyePos, gun.pose);

	// Glowing bits, in room space, added on top. The line is hidden by the gun where it
	// passes behind it; the round spots (flash, dot) are always seen.
	static std::vector<GlowVertex> line, spots;
	line.clear();
	spots.clear();
	const glm::vec3 muzzle = glm::vec3(gun.pose * glm::vec4(GunMuzzle, 1.f));
	if (gun.aimLine)
	{
		const glm::vec3 dir = gun.aimPoint - muzzle;
		const float len = glm::length(dir);
		if (len > 0.05f)
		{
			const glm::vec3 fwd = dir / len;
			glm::vec3 side = glm::cross(fwd, eyePos - muzzle);
			side = glm::length(side) < 1e-6f ? glm::vec3(0.0008f, 0, 0) : glm::normalize(side) * 0.0008f;
			// fades out along the way: it guides without painting over the scene
			const glm::vec4 start { 1.f, 0.08f, 0.05f, 0.55f }, end { 1.f, 0.08f, 0.05f, 0.f };
			const glm::vec3 to = muzzle + fwd * std::min(len, 3.f);
			const GlowVertex a { muzzle - side, start, {} }, b { muzzle + side, start, {} };
			const GlowVertex c { to + side, end, {} }, d { to - side, end, {} };
			for (const GlowVertex& v : { a, b, c, a, c, d })
				line.push_back(v);
		}
	}
	if (gun.flash > 0.f)
	{
		const float size = 0.025f + 0.03f * gun.flash;
		const glm::vec3 at = glm::vec3(gun.pose * glm::vec4(GunMuzzle + glm::vec3(0, 0, -0.02f), 1.f));
		billboard(spots, at, eyePos, size, glm::vec4(1.f, 0.75f, 0.3f, gun.flash));
		billboard(spots, at, eyePos, size * 0.45f, glm::vec4(1.f, 1.f, 0.85f, gun.flash));
	}
	if (gun.aimDot)
	{
		// about 0.6 degrees across wherever it lands
		const float size = glm::length(gun.aimPoint - eyePos) * 0.0055f;
		billboard(spots, gun.aimPoint, eyePos, size, glm::vec4(1.f, 0.12f, 0.06f, 0.95f));
		billboard(spots, gun.aimPoint, eyePos, size * 0.4f, glm::vec4(1.f, 0.85f, 0.7f, 1.f));
	}
	glcache.UseProgram(glowProgram);
	glUniformMatrix4fv(gMvp, 1, GL_FALSE, &viewProj[0][0]);
	glcache.Enable(GL_BLEND);
	glcache.BlendFunc(GL_SRC_ALPHA, GL_ONE);
	glcache.DepthMask(GL_FALSE);
	glEnableVertexAttribArray(VERTEX_POS_ARRAY);
	glEnableVertexAttribArray(VERTEX_COL_BASE_ARRAY);
	glEnableVertexAttribArray(VERTEX_UV_ARRAY);
	glUniform1f(gSpot, 0.f);
	drawGlow(line);
	glUniform1f(gSpot, 1.f);
	glcache.Disable(GL_DEPTH_TEST);
	drawGlow(spots);
	glDisableVertexAttribArray(VERTEX_POS_ARRAY);
	glDisableVertexAttribArray(VERTEX_COL_BASE_ARRAY);
	glDisableVertexAttribArray(VERTEX_UV_ARRAY);
	glcache.DepthMask(GL_TRUE);
	glcache.Disable(GL_BLEND);
}

void termGun()
{
	if (modelProgram != 0)
		glcache.DeleteProgram(modelProgram);
	if (glowProgram != 0)
		glcache.DeleteProgram(glowProgram);
	if (modelVbo != 0)
		glDeleteBuffers(1, &modelVbo);
	if (modelIbo != 0)
		glDeleteBuffers(1, &modelIbo);
	modelProgram = glowProgram = 0;
	modelVbo = modelIbo = 0;
	mMvp = mModel = mEye = mColor = mSpecular = mShininess = mEmissive = -1;
	gMvp = gSpot = -1;
}

}
