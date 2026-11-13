#version 460 core
#include <flutter/runtime_effect.glsl>
uniform vec2 uSize;
uniform float uPhase;
uniform float uLevel;
uniform float uActive;
out vec4 fragColor;
void main() {
  vec2 p = FlutterFragCoord().xy - uSize * 0.5;
  float base = min(uSize.x, uSize.y) * 0.34;
  float d = length(p);
  float band = smoothstep(base * 0.85, base, d) * (1.0 - smoothstep(base * 1.5, base * 1.75, d));
  float carrier = sin(d * 0.15 - uPhase * 6.2831853) * 0.5 + 0.5;
  float echo = sin(length(p - vec2(base * 0.16, 0.0)) * 0.18 + uPhase * 6.2831853) * 0.5 + 0.5;
  vec2 cell = floor(FlutterFragCoord().xy / 4.0);
  float threshold = fract(dot(mod(cell, 4.0), vec2(0.3125, 0.5625)));
  float density = (carrier * 0.65 + echo * 0.35) * (0.10 + min(uLevel * 2.0, 0.8) * uActive);
  float dotMask = (1.0 - step(1.4, mod(FlutterFragCoord().x, 4.0))) * (1.0 - step(1.4, mod(FlutterFragCoord().y, 4.0)));
  float alpha = step(threshold, density) * dotMask * band * (0.18 + uLevel * 0.65);
  fragColor = vec4(vec3(alpha), alpha);
}
