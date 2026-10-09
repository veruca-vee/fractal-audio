#version 330

uniform vec2 u_resolution;
uniform vec2 u_c;          // Julia constant, driven by the current mode's path
uniform int u_power;       // d in z^d + c: 2, 3, 5 or 8
uniform int u_formula;     // which map to iterate; see the FORMULA_* below
uniform vec2 u_p;          // mode 6 only: the Phoenix map's memory coefficient
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

// The voice's own shape. An orbit trap: as each point's orbit is iterated,
// its closest approach to a rose curve r = R|cos(k.theta)| is remembered,
// and points whose orbits brush the curve are tinted. The shape is drawn by
// the fractal's own dynamics rather than laid over them, so it grows out of
// the structure and, at weight 0, costs the picture nothing at all.
uniform float u_trap_weight;  // 0 = no voice, nothing drawn
uniform float u_trap_k;       // petals; fractional, so it can morph
uniform float u_trap_rot;     // the contour turns it
uniform float u_trap_radius;  // ...and sizes it
uniform float u_trap_sharp;   // how tightly the tint hugs the curve
uniform float u_trap_hue;     // its own place in the palette

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

// How many iterations the voice's trap looks at, and it is the single
// biggest cost in the shape: 6 iterations render a singing frame in 19.4 ms
// against 35.4 ms at 12. (One measurement here claimed the two were within
// half a percent. They are not -- that run interleaved two programs in one
// context and the second was reading the first's thermal state. The test
// caught the revert, which is the only reason this comment is right.)
// The early orbit is what spreads a point across the plane, so 6 still
// draws nearly all of the shape.
const int TRAP_ITERS = 6;

// Must match phi_paths.FORMULA_*.
const int FORMULA_POWER = 0;    // z^d + c            (modes 1-5)
const int FORMULA_PHOENIX = 1;  // z^d + c + p*z_prev (mode 6)
const int FORMULA_TRICORN = 2;  // conj(z)^d + c      (mode 7)
const int FORMULA_SHIP = 3;     // (|x| + i|y|)^d + c (mode 7)

// atan to about 0.01 rad, which is far inside the width of the band the
// trap lights up. Worth 8% of the singing frame on this GPU (21.1 ms down
// to 19.4 ms) -- real, but not the rescue an earlier measurement here
// claimed. That measurement was taken with a wall clock against a driver
// that was eliding the draws entirely, and it was wrong by a wide margin.
float fast_atan2(float y, float x) {
    float ax = abs(x), ay = abs(y);
    float a = min(ax, ay) / (max(ax, ay) + 1e-12);
    float s = a * a;
    float r = ((-0.0464964749 * s + 0.15931422) * s - 0.327622764) * s * a + a;
    if (ay > ax) r = 1.57079637 - r;
    if (x < 0.0) r = 3.14159274 - r;
    return (y < 0.0) ? -r : r;
}

// Approximate distance from z to the rose r = R|cos(k.theta)|. Not exact --
// the true distance to a rose has no closed form -- but the error is radial
// and smooth, which is all an orbit trap needs.
float rose_distance(vec2 z) {
    float a = fast_atan2(z.y, z.x) + u_trap_rot;
    return abs(length(z) - u_trap_radius * abs(cos(u_trap_k * a)));
}

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

// z raised to u_power (2, 3, 5 or 8) by repeated squaring, so the degree
// costs at most four multiplies however the modes set it.
vec2 cpow_d(vec2 z) {
    vec2 z2 = cmul(z, z);
    if (u_power == 2) return z2;
    if (u_power == 3) return cmul(z2, z);
    if (u_power == 5) return cmul(cmul(z2, z2), z);
    vec2 z4 = cmul(z2, z2);
    return cmul(z4, z4);
}

// The escape loop, with the map it iterates pasted in: testing u_formula
// inside the loop instead costs about a third of the frame rate on a small
// GPU (measured: 64 -> 45 fps at 1024x576 on Intel UHD 600), so each family
// gets its own copy of the loop with no test in it at all.
//
// SETTLED is the extra condition for calling a repeat of z a settled orbit.
// It's just `true` for the maps whose whole state is z; the Phoenix map's
// state is the pair (z, z_prev), so there a repeat of z on its own isn't a
// cycle and would cut the run short and misread it as interior.
#define ESCAPE_LOOP(STEP, SETTLED, TRAP)                                       \
    for (int i = 0; i < 1000; i++) {                                           \
        if (i >= max_iter) break;                                              \
        STEP;                                                                  \
        float d2 = dot(z, z);                                                  \
        trap = min(trap, d2);                                                  \
        if (iter < 12) axis_trap = min(axis_trap, min(abs(z.x), abs(z.y)));    \
        if (iter < TRAP_ITERS) { TRAP }                                        \
        if (d2 > 256.0) break;                                                 \
        iter++;                                                                \
        /* Interior points settle into a cycle; once z revisits a saved     */ \
        /* point, stop instead of burning the remaining iterations.         */ \
        if ((iter & 7) == 0) {                                                 \
            vec2 dz = z - saved;                                               \
            if (dot(dz, dz) < 1e-7 && (SETTLED)) { iter = max_iter; break; }   \
            if ((iter & 31) == 0) saved = z;                                   \
        }                                                                      \
    }

// The four maps, instantiated with or without the voice's trap above.
#define ALL_FORMULAS(TRAP)                                                     \
    if (u_formula == FORMULA_POWER) {                                          \
        ESCAPE_LOOP(z = cpow_d(z) + u_c, true, TRAP)                           \
    } else if (u_formula == FORMULA_TRICORN) {                                 \
        ESCAPE_LOOP(z = cpow_d(vec2(z.x, -z.y)) + u_c, true, TRAP)             \
    } else if (u_formula == FORMULA_SHIP) {                                    \
        ESCAPE_LOOP(z = cpow_d(abs(z)) + u_c, true, TRAP)                      \
    } else {                                                                   \
        ESCAPE_LOOP(vec2 zn = cpow_d(z) + u_c + cmul(u_p, z_prev);             \
                    z_prev = z; z = zn,                                        \
                    dot(z - z_prev, z - z_prev) < 1e-7, TRAP)                  \
    }

vec3 julia(vec2 z, int max_iter) {
    float r0 = log(length(z) + 0.2);
    int iter = 0;
    float trap = 1e9;
    float axis_trap = 1e9;
    float rose_trap = 1e9;
    vec2 saved = z;
    // Only the Phoenix map keeps a previous z; the others never read it.
    vec2 z_prev = vec2(0.0);
    // Whether the voice's trap is gathered is decided once, out here, not
    // per iteration. rose_distance costs an atan, and leaving it in the loop
    // to be multiplied by a zero weight later cost a quarter of the frame
    // rate on every silent frame -- measured, 63 fps fell to 48 with nothing
    // on screen to show for it. Most music is not wall-to-wall singing.
    if (u_trap_weight > 0.001) {
        ALL_FORMULAS(rose_trap = min(rose_trap, rose_distance(z));)
    } else {
        ALL_FORMULAS(;)
    }

    // How strongly this point's orbit brushed the voice's curve. Bounded,
    // and zero whenever the voice is: the fractal is then exactly itself.
    float voice = u_trap_weight * exp(-rose_trap * u_trap_sharp);
    vec3 voice_col = palette(u_trap_hue) * 1.35;

    if (iter >= max_iter) {
        // Interior: how close the orbit passed to the axes draws stalks and
        // petals inside the set, so it never reads as one flat dark blob.
        float stalk = log(axis_trap + 1e-4);
        vec3 inner = palette(stalk * 0.18 + sqrt(trap) * 0.5 + u_mid * 0.4 + u_time * 0.015 + 0.5);
        float glow = clamp(0.3 - stalk * 0.08, 0.0, 1.0);
        // Hats ripple the stalks.
        inner *= 1.0 + 0.45 * u_hat * sin(stalk * 7.0);
        inner *= (0.3 + 0.25 * u_amp + 0.3 * glow + 0.35 * u_kick);
        // mix, not add: the shape can reach the eye without ever blowing
        // the picture out, however loud the singing gets.
        return mix(inner, voice_col, min(voice, 0.8));
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
    col = col * (val + 0.3 * u_kick) * (1.0 + 0.4 * u_hat * contour);
    return mix(col, voice_col, min(voice, 0.8));
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
