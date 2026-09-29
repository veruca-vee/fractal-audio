#version 330

uniform vec2 u_resolution;
uniform vec2 u_c;          // Julia constant, driven by the current mode's path
uniform int u_power;       // d in z^d + c: 2, 3, 5 or 8
uniform vec2 u_center;     // plane point at the middle of the view
uniform float u_zoom;      // view scale; smaller = zoomed in
uniform float u_amp;       // overall audio amplitude, 0..1
uniform float u_bass;      // low-band energy, 0..1
uniform float u_mid;       // mid-band energy, 0..1
uniform float u_treble;    // high-band energy, 0..1
uniform float u_kick;      // 1 on a kick drum hit, decaying to 0
uniform float u_snare;     // 1 on a snare/backbeat hit, decaying to 0
uniform float u_hat;       // 1 on a hi-hat hit, decaying to 0
uniform float u_hue;       // palette phase shift: the notes' pitch and the snare steps
uniform int u_max_iter;
uniform float u_time;

uniform float u_spin;      // view rotation, radians

// Vocal split: the plane is mirror-folded into n wedges, each holding a
// smaller copy pushed out into a ring, then twisted into spiral arms.
// Two fold counts are crossfaded so n can change without popping.
uniform float u_fold_a;
uniform float u_fold_b;
uniform float u_fold_mix;
uniform float u_twist;

// Cosine palette: a + b * cos(2pi * (c * t + d))
uniform vec3 u_pal_a;
uniform vec3 u_pal_b;
uniform vec3 u_pal_c;
uniform vec3 u_pal_d;

out vec4 fragColor;

const float TAU = 6.28318530718;

vec3 palette(float t) {
    return u_pal_a + u_pal_b * cos(TAU * (u_pal_c * t + u_pal_d + u_hue));
}

vec2 split(vec2 z, float n) {
    float r = length(z);
    float a = atan(z.y, z.x) + u_twist * log(r + 1e-4);
    if (n > 1.5) {
        float s = TAU / n;
        a = abs(mod(a + 0.5 * s, s) - 0.5 * s);
        // More wedges -> smaller copies, set further out, so they fit.
        float ring = 0.35 + 0.1 * n;
        float scale = 1.0 + 0.3 * (n - 1.0);
        return (r * vec2(cos(a), sin(a)) - vec2(ring, 0.0)) * scale;
    }
    return r * vec2(cos(a), sin(a));
}

vec2 cmul(vec2 a, vec2 b) {
    return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

vec3 julia(vec2 z, int max_iter) {
    float r0 = log(length(z) + 0.2);
    int iter = 0;
    float trap = 1e9;
    float axis_trap = 1e9;
    vec2 saved = z;
    for (int i = 0; i < 1000; i++) {
        if (i >= max_iter) break;
        vec2 z2 = cmul(z, z);
        if (u_power == 2) z = z2;
        else if (u_power == 3) z = cmul(z2, z);
        else if (u_power == 5) z = cmul(cmul(z2, z2), z);
        else { vec2 z4 = cmul(z2, z2); z = cmul(z4, z4); }
        z += u_c;
        float d2 = dot(z, z);
        trap = min(trap, d2);
        if (iter < 12) axis_trap = min(axis_trap, min(abs(z.x), abs(z.y)));
        if (d2 > 256.0) break;
        iter++;
        // Interior points settle into a cycle; once z revisits a saved
        // point, stop instead of burning the remaining iterations.
        if ((iter & 7) == 0) {
            vec2 dz = z - saved;
            if (dot(dz, dz) < 1e-7) { iter = max_iter; break; }
            if ((iter & 31) == 0) saved = z;
        }
    }

    if (iter >= max_iter) {
        // Interior: how close the orbit passed to the axes draws stalks and
        // petals inside the set, so it never reads as one flat dark blob.
        float stalk = log(axis_trap + 1e-4);
        vec3 inner = palette(stalk * 0.18 + sqrt(trap) * 0.5 + u_mid * 0.4 + u_time * 0.015 + 0.5);
        float glow = clamp(0.3 - stalk * 0.08, 0.0, 1.0);
        // Hats ripple the stalks.
        inner *= 1.0 + 0.45 * u_hat * sin(stalk * 7.0);
        return inner * (0.3 + 0.25 * u_amp + 0.3 * glow + 0.35 * u_kick);
    }

    // Smooth iteration count for band-free colouring.
    float smooth_iter = float(iter) + 1.0 - log(log2(dot(z, z)) * 0.5) / log(float(u_power));
    float t = smooth_iter / float(max_iter);

    // Hue walks with escape speed and mid-band energy; saturation pulses
    // with treble; value (brightness) pulses with bass + overall amplitude.
    vec3 col = palette(log2(smooth_iter + 1.0) * 0.3 + r0 * 0.25 + u_mid * 0.4 + u_time * 0.015);
    float grey = dot(col, vec3(0.299, 0.587, 0.114));
    col = mix(vec3(grey), col, 0.6 + 0.4 * u_treble);
    float val = 0.45 + 0.55 * clamp(t * 6.0 + u_bass * 0.4 + u_amp * 0.2, 0.0, 1.0);
    // Hats flick fine contour lines through the escape-time bands.
    float contour = cos(smooth_iter * 2.4);
    return col * (val + 0.3 * u_kick) * (1.0 + 0.4 * u_hat * contour);
}

// z is relative to the view center, so the vocal fold splits around the
// middle of the screen.
vec3 shade(vec2 z) {
    if (u_fold_mix <= 0.001 || u_fold_mix >= 0.999) {
        return julia(u_center + split(z, u_fold_mix < 0.5 ? u_fold_a : u_fold_b), u_max_iter);
    }
    return mix(julia(u_center + split(z, u_fold_a), u_max_iter),
               julia(u_center + split(z, u_fold_b), u_max_iter), u_fold_mix);
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_resolution) / u_resolution.y;
    float c = cos(u_spin), s = sin(u_spin);
    vec2 z = mat2(c, s, -s, c) * uv * 2.0 * u_zoom;
    vec3 col = shade(z);
    // A snare hit briefly flips the picture toward its negative.
    col = mix(col, vec3(1.0) - col, 0.4 * u_snare);
    fragColor = vec4(col, 1.0);
}
