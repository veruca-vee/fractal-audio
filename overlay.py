"""The now-playing overlay: where you are in the track, and the transport.

Two parts, deliberately unequal:

- The edge light is always on while something is playing. It traces the
  screen border clockwise from the top-left corner, lit behind a brighter
  head that sits at the current position, so the whole frame is the progress
  bar and nothing is laid over the fractal. It is drawn additively and kept
  dim: at a glance it reads as the picture having an edge, and you only
  notice it is moving when you look for it.

- The buttons are not on until you ask. A click anywhere brings them up, a
  click anywhere else puts them away again, and a click on one of them does
  what it says and leaves them up. That is mpv's on-screen controller
  without the auto-hide timer: on a fractal you are watching for minutes at
  a time, something that fades in and out by itself is a distraction, and
  the mouse is otherwise doing nothing here.

The textures are built once (the glyphs) or only when their text changes
(the title line), never per frame -- at 60 fps the budget for all of this is
what's left of 16 ms after the fractal, which is not much.
"""

import math

import moderngl
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

GLYPH_PX = 128          # the glyph textures are built this big and scaled down
BUTTON_FRACTION = 0.062  # button diameter as a fraction of window height
BUTTON_MIN_PX = 44       # ...but never smaller than a comfortable target
BUTTON_GAP = 0.35        # gap between buttons, as a fraction of the diameter
BOTTOM_MARGIN = 0.10     # button row above the bottom edge, fraction of height
FADE_SECONDS = 0.18      # the controls fade rather than pop

EDGE_WIDTH_PX = 6.0      # how far the edge light reaches in from the border
EDGE_TRACK = 0.03        # the whole border, so it reads as a ring not an arc
EDGE_BASE = 0.10         # brightness of the part already played
EDGE_HEAD = 0.70          # brightness of the moving head
EDGE_HEAD_PX = 80.0     # how long the head's tail is, in pixels of perimeter
EDGE_CORE_PX = 11.0      # the head's own size, so it's visible at 0:00 too
EDGE_PAUSED = 0.45       # the light dims to this while paused

# A plain textured quad, used for every image this module and the toasts draw.
QUAD_VERT = """#version 330
uniform vec4 u_rect;  // x0, y0, x1, y1 in NDC
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    vec2 p = mix(u_rect.xy, u_rect.zw, v_uv);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

QUAD_FRAG = """#version 330
uniform sampler2D u_tex;
uniform float u_alpha;
in vec2 v_uv;
out vec4 fragColor;
void main() {
    vec4 c = texture(u_tex, v_uv);
    fragColor = vec4(c.rgb, c.a * u_alpha);
}
"""

# The edge light covers the whole frame, so it gets a vertex shader of its
# own rather than the quad's: that one positions itself from a u_rect, and an
# unset u_rect collapses the quad to a point and draws nothing at all.
FULLSCREEN_VERT = """#version 330
in vec2 in_position;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

# The edge light. Every pixel works out how far round the perimeter it sits,
# clockwise from the top-left corner, and lights up if the track has got
# that far. Drawn additively, so it brightens the fractal rather than
# covering it.
EDGE_FRAG = """#version 330
uniform vec2 u_resolution;
uniform float u_progress;   // 0..1 through the track
uniform float u_alpha;      // overall strength; 0 draws nothing
uniform vec3 u_tint;
out vec4 fragColor;

const float WIDTH = """ + f"{EDGE_WIDTH_PX}" + """;
const float TRACK = """ + f"{EDGE_TRACK}" + """;
const float BASE = """ + f"{EDGE_BASE}" + """;
const float HEAD = """ + f"{EDGE_HEAD}" + """;
const float HEAD_PX = """ + f"{EDGE_HEAD_PX}" + """;
const float CORE_PX = """ + f"{EDGE_CORE_PX}" + """;

void main() {
    vec2 p = gl_FragCoord.xy;
    vec2 r = u_resolution;

    float d_left = p.x, d_right = r.x - p.x;
    float d_bottom = p.y, d_top = r.y - p.y;
    float dist = min(min(d_left, d_right), min(d_bottom, d_top));

    // Arc length from the top-left corner, clockwise. The four cases agree
    // at the corners, so the light runs round without a seam.
    float t;
    if (d_top <= dist)         t = p.x;
    else if (d_right <= dist)  t = r.x + (r.y - p.y);
    else if (d_bottom <= dist) t = r.x + r.y + (r.x - p.x);
    else                       t = 2.0 * r.x + r.y + p.y;

    float total = 2.0 * (r.x + r.y);
    float head_at = u_progress * total;

    // The whole border carries a faint track; the played part of it is
    // brighter, and the head is the bright point you can actually follow.
    float tail = head_at - t;
    // A small core centred on the head, so there's a point to see even at
    // 0:00 with nothing played yet, and a longer tail behind it only. The
    // tail has to stay one-sided: let it reach in front of the head as well
    // and the unplayed border becomes the brightest thing on screen, which
    // is the picture exactly backwards.
    float lit = TRACK + HEAD * exp(-abs(tail) / CORE_PX);
    if (tail >= 0.0) {
        lit += BASE + HEAD * 0.75 * exp(-tail / HEAD_PX);
    }
    float falloff = exp(-dist / WIDTH);

    fragColor = vec4(u_tint * lit * falloff * u_alpha, 1.0);
}
"""


def _glyph(kind, size=GLYPH_PX):
    """One transport button as an RGBA image: a dark disc with a white mark.

    Drawn at 4x and downsampled, because PIL has no antialiasing of its own
    and a jagged triangle is very visible against a fractal.
    """
    s = size * 4
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, s - 1, s - 1), fill=(0, 0, 0, 150), outline=(255, 255, 255, 90),
              width=max(1, s // 64))

    w = s * 0.30          # mark half-width
    h = s * 0.30          # mark half-height
    cx = cy = s / 2
    white = (255, 255, 255, 235)
    if kind == "play":
        # Nudged right by a hair: a triangle centered on its bounding box
        # looks off-center inside a circle.
        d.polygon([(cx - w * 0.75 + s * 0.03, cy - h), (cx - w * 0.75 + s * 0.03, cy + h),
                   (cx + w, cy)], fill=white)
    elif kind == "pause":
        bar = w * 0.42
        d.rectangle((cx - w * 0.78, cy - h, cx - w * 0.78 + bar, cy + h), fill=white)
        d.rectangle((cx + w * 0.78 - bar, cy - h, cx + w * 0.78, cy + h), fill=white)
    else:  # "prev" / "next": two triangles against a bar
        sign = -1 if kind == "prev" else 1
        bar = w * 0.22
        tri = w * 0.62
        for k in (0, 1):
            tip = cx + sign * (w - k * tri)
            back = tip - sign * tri
            d.polygon([(back, cy - h), (back, cy + h), (tip, cy)], fill=white)
        edge = cx + sign * (w + bar * 0.9)
        d.rectangle((min(edge, edge - sign * bar), cy - h,
                     max(edge, edge - sign * bar), cy + h), fill=white)

    img = img.resize((size, size), Image.LANCZOS)
    return img.transpose(Image.FLIP_TOP_BOTTOM)  # GL origin is bottom-left


def _text_image(text, px=20):
    """The title line, as a pill like the toasts wear."""
    font = ImageFont.truetype(FONT, px)
    l, t, r, b = font.getbbox(text)
    pad_x, pad_y = 16, 9
    w, h = (r - l) + 2 * pad_x, (b - t) + 2 * pad_y
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w - 1, h - 1), radius=h // 2, fill=(0, 0, 0, 150))
    d.text((pad_x - l, pad_y - t), text, font=font, fill=(255, 255, 255, 235))
    return img.transpose(Image.FLIP_TOP_BOTTOM)


def clock(seconds):
    """1:23. Negative or missing reads as 0:00."""
    seconds = max(0.0, float(seconds or 0.0))
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


class NowPlayingOverlay:
    """The edge light and the transport buttons, over an already-drawn frame.

    The caller owns the player; this only reads it. `click` is given window
    coordinates with the origin at the top left, the way the window hands
    them over, and returns the name of whatever it hit.
    """

    BUTTONS = ("prev", "play_pause", "next")

    def __init__(self, ctx, quad_vbo, tint=(0.55, 0.75, 1.0)):
        self.ctx = ctx
        self.tint = tint
        self.visible = False
        self.shown_at = 0.0
        self.hidden_at = -1e9
        self._text = None
        self._text_tex = None
        self._text_size = (1, 1)

        self.edge_prog = ctx.program(vertex_shader=FULLSCREEN_VERT, fragment_shader=EDGE_FRAG)
        self.edge_vao = ctx.simple_vertex_array(self.edge_prog, quad_vbo, "in_position")
        self.quad_prog = ctx.program(vertex_shader=QUAD_VERT, fragment_shader=QUAD_FRAG)
        self.quad_vao = ctx.simple_vertex_array(self.quad_prog, quad_vbo, "in_position")

        self.glyphs = {}
        for kind in ("prev", "play", "pause", "next"):
            img = _glyph(kind)
            tex = ctx.texture(img.size, 4, img.tobytes())
            tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.glyphs[kind] = tex

    # ---- visibility --------------------------------------------------------

    def show(self, now):
        if not self.visible:
            self.visible = True
            self.shown_at = now

    def hide(self, now):
        if self.visible:
            self.visible = False
            self.hidden_at = now

    def toggle(self, now):
        self.hide(now) if self.visible else self.show(now)

    def _alpha(self, now):
        """Fade in while shown, out for a moment after being dismissed."""
        if self.visible:
            return min(1.0, (now - self.shown_at) / FADE_SECONDS)
        return max(0.0, 1.0 - (now - self.hidden_at) / FADE_SECONDS)

    # ---- layout and hit-testing -------------------------------------------

    def layout(self, size):
        """name -> (x0, y0, x1, y1) in window pixels, origin top left."""
        w, h = size
        d = max(BUTTON_MIN_PX, h * BUTTON_FRACTION)
        gap = d * BUTTON_GAP
        cy = h * (1.0 - BOTTOM_MARGIN) - d / 2
        span = 3 * d + 2 * gap
        x = w / 2 - span / 2
        rects = {}
        for name in self.BUTTONS:
            rects[name] = (x, cy, x + d, cy + d)
            x += d + gap
        return rects

    def click(self, x, y, size, now):
        """Handle a press at window coordinates (origin top left).

        Returns the button hit ("prev", "play_pause", "next"), or None for a
        click that only showed or dismissed the controls.
        """
        if not self.visible:
            self.show(now)
            return None
        for name, (x0, y0, x1, y1) in self.layout(size).items():
            cx, cy, r = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2
            if math.hypot(x - cx, y - cy) <= r:   # round buttons, round targets
                return name
        self.hide(now)
        return None

    # ---- drawing -----------------------------------------------------------

    def _draw_tex(self, tex, rect, alpha):
        self.quad_prog["u_rect"].value = rect
        self.quad_prog["u_alpha"].value = alpha
        tex.use(0)
        self.quad_prog["u_tex"].value = 0
        self.quad_vao.render(moderngl.TRIANGLE_STRIP)

    def _ndc(self, rect, size):
        w, h = size
        x0, y0, x1, y1 = rect
        return (2 * x0 / w - 1, 1 - 2 * y1 / h, 2 * x1 / w - 1, 1 - 2 * y0 / h)

    def draw(self, player, now, size, buffer_size):
        """Draw the edge light, and the controls if they're up.

        Expects to be called with the window bound and the scene already in
        it; leaves blending off, the way it found it.
        """
        if player is None or not player.alive:
            return

        duration, position = player.duration, player.position
        if duration > 0:
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = moderngl.ONE, moderngl.ONE   # additive: a light
            self.edge_prog["u_resolution"].value = buffer_size
            self.edge_prog["u_progress"].value = min(1.0, max(0.0, position / duration))
            self.edge_prog["u_alpha"].value = EDGE_PAUSED if player.paused else 1.0
            self.edge_prog["u_tint"].value = self.tint
            self.edge_vao.render(moderngl.TRIANGLE_STRIP)
            self.ctx.blend_func = moderngl.DEFAULT_BLENDING
            self.ctx.disable(moderngl.BLEND)

        alpha = self._alpha(now)
        if alpha <= 0.001:
            return

        line = f"{player.title()}    {clock(position)} / {clock(duration)}"
        if line != self._text:
            self._text = line
            if self._text_tex is not None:
                self._text_tex.release()
            img = _text_image(line)
            self._text_tex = self.ctx.texture(img.size, 4, img.tobytes())
            self._text_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._text_size = img.size

        self.ctx.enable(moderngl.BLEND)
        rects = self.layout(size)
        for name, rect in rects.items():
            kind = name
            if name == "play_pause":
                kind = "play" if player.paused else "pause"
            self._draw_tex(self.glyphs[kind], self._ndc(rect, size), alpha)

        # The title sits centered above the row.
        w, h = size
        tw, th = self._text_size
        top = min(r[1] for r in rects.values())
        text_rect = (w / 2 - tw / 2, top - th - 0.02 * h, w / 2 + tw / 2, top - 0.02 * h)
        self._draw_tex(self._text_tex, self._ndc(text_rect, size), alpha)
        self.ctx.disable(moderngl.BLEND)

    def release(self):
        for tex in self.glyphs.values():
            tex.release()
        if self._text_tex is not None:
            self._text_tex.release()
