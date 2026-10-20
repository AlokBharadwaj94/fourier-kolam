"""
kolam_studio.py
===============
Interactive panels (ipywidgets) for making a Fourier kolam from any image.

    studio = Studio("photo.jpg")
    studio.image_panel()     # 1  load or upload an image
    studio.spacing_panel()   # 2  sample spacing in frequency space, spectrum behind the dots
    studio.mask_panel()      # 3  outline: predefined shapes or your own mask image
    studio.style_panel()     # 4  palette, background, transparency, pulli, petals
    studio.export_panel()    # 5-6  pdf / eps / png at a chosen size and dpi

All the computation lives in kolam_fourier.py; this file only wires widgets to it.
"""
import io
import os
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as W
from IPython.display import display, clear_output, FileLink
from PIL import Image

import kolam_fourier as kf

SHAPE_NAMES = ["diamond", "circle", "quatrefoil", "four-point star", "eight-point star",
               "mandelbrot", "custom mask"]


def _uploaded_image(change):
    """First file of an ipywidgets 8 FileUpload value, as a PIL image."""
    files = change["new"]
    if not files:
        return None, None
    f = files[0]
    return Image.open(io.BytesIO(bytes(f["content"]))), f["name"]


class Studio:
    def __init__(self, image_path=None, size=512):
        kf.dark_style()
        self.size = size
        self.img = self.sp = self.lat = None
        self.mask = None
        self.K = None                  # the built kolam
        self._dirty = True             # structure changed since the last build
        self.preview = None
        self._build_widgets()
        if image_path:
            self.path.value = image_path
            self._load_path()

    # ------------------------------------------------------------------ widgets
    def _build_widgets(self):
        sty = {"description_width": "150px"}
        lay = W.Layout(width="520px")
        # 1 image
        self.path = W.Text(description="image file", style=sty, layout=lay)
        self.load_btn = W.Button(description="Load", button_style="primary")
        self.upload = W.FileUpload(accept="image/*", multiple=False, description="or upload")
        self.img_out = W.Output()
        self.load_btn.on_click(lambda _: self._load_path())
        self.upload.observe(self._on_upload, "value")
        # 2 spacing
        self.du = W.FloatSlider(description="spacing across (du)", min=0.5, max=25, step=0.001, value=4.6,
                                continuous_update=False, style=sty, layout=lay, readout_format=".3f")
        self.dv = W.FloatSlider(description="spacing up (dv)", min=0.5, max=25, step=0.001, value=5.4,
                                continuous_update=False, style=sty, layout=lay, readout_format=".3f")
        self.R = W.FloatSlider(description="outline size (samples)", min=4, max=40, step=0.25, value=20,
                               continuous_update=False, style=sty, layout=lay)
        self.div_btns = [W.Button(description=f"lattice ÷ {k}", layout=W.Layout(width="110px")) for k in (1, 2, 3, 4)]
        for k, b in zip((1, 2, 3, 4), self.div_btns):
            b.on_click(lambda _, k=k: self._set_division(k))
        self.spec_out = W.Output()
        # 3 mask
        self.shape = W.Dropdown(description="outline", options=SHAPE_NAMES, value="diamond", style=sty, layout=lay)
        self.mask_up = W.FileUpload(accept="image/*", multiple=False, description="mask image")
        self.invert = W.Checkbox(description="invert mask (light = inside)", value=False)
        self.mask_out = W.Output()
        self.mask_up.observe(self._on_mask, "value")
        # 4 style
        self.palette = W.Dropdown(description="palette", options=list(kf.PALETTES), value="red oxide floor",
                                  style=sty, layout=lay)
        self.c_bg = W.ColorPicker(description="background", style=sty)
        self.c_line = W.ColorPicker(description="lines", style=sty)
        self.c_lv = [W.ColorPicker(description=f"level {k}", style=sty) for k in (1, 2, 3, 4)]
        self.c_dot = W.ColorPicker(description="pulli", value="#FFFFFF", style=sty)
        self.transparent = W.Checkbox(description="transparent background", value=False)
        self.pulli = W.Checkbox(description="pulli (dots)", value=True)
        self.petals = W.Checkbox(description="petals on the edge", value=True)
        self.glow = W.Checkbox(description="glow around 3s and 4s", value=True)
        self.dot_size = W.FloatSlider(description="pulli size", min=.05, max=.3, step=.01, value=.13,
                                      continuous_update=False, style=sty, layout=lay)
        self.tip = W.FloatSlider(description="petal length", min=1.0, max=1.8, step=.05, value=1.25,
                                 continuous_update=False, style=sty, layout=lay)
        self.method = W.Dropdown(description="gate rule", style=sty, layout=lay,
                                 options=[("level growth (4s first)", "growth"), ("shell sweep (from DC)", "sweep")])
        self.seed = W.IntText(description="random seed", value=2, style=sty)
        self.render_btn = W.Button(description="Render preview", button_style="primary")
        self.style_out = W.Output()
        self.palette.observe(lambda _: self._apply_palette(), "value")
        self.render_btn.on_click(lambda _: self._render_preview())
        self._apply_palette()
        # 5-6 export
        self.fmt = W.ToggleButtons(description="format", options=["pdf", "eps", "png"], value="pdf", style=sty)
        self.width_mm = W.FloatText(description="width (mm)", value=118.0, style=sty)
        self.dpi = W.IntText(description="dpi", value=600, style=sty)
        self.fname = W.Text(description="file name", value="kolam", style=sty)
        self.folder = W.Text(description="folder", value="exports", style=sty)
        self.export_btn = W.Button(description="Export", button_style="success")
        self.export_out = W.Output()
        self.export_btn.on_click(lambda _: self._export())
        # structure changes -> rebuild; redraw the live previews
        for w in (self.du, self.dv, self.R):
            w.observe(lambda _: self._changed(spacing=True), "value")
        for w in (self.shape, self.invert):
            w.observe(lambda _: self._changed(spacing=True), "value")
        for w in (self.method, self.seed):
            w.observe(lambda _: self._changed(), "value")

    # ------------------------------------------------------------------ state
    def _load_image(self, pil, name):
        tmp = "_studio_input.png"; pil.convert("RGB").save(tmp)
        self.img = kf.load_image(tmp, self.size)
        with self.img_out:
            clear_output(wait=True)
            print(f"loading {name} … computing the Fourier transform")
        self.sp = kf.compute_spectrum(self.img, 8)
        self.lat = kf.estimate_lattice(self.sp)
        self._set_division(3, redraw=False)
        self._dirty = True
        self._draw_image()
        self._draw_spectrum(); self._draw_mask()

    def _load_path(self):
        if self.path.value and os.path.exists(self.path.value):
            self._load_image(Image.open(self.path.value), self.path.value)
        else:
            with self.img_out:
                clear_output(); print("file not found:", self.path.value)

    def _on_upload(self, change):
        pil, name = _uploaded_image(change)
        if pil is not None:
            self._load_image(pil, name)

    def _on_mask(self, change):
        pil, name = _uploaded_image(change)
        if pil is not None:
            self.mask = pil
            self.shape.value = "custom mask"
            self._changed(spacing=True)

    def _set_division(self, k, redraw=True):
        if self.lat is None:
            return
        a, b = self.lat["a_star_native"], self.lat["b_star_native"]
        for s, v in ((self.du, a / k), (self.dv, b / k)):
            s.max = max(s.max, v * 1.2); s.value = v
        if redraw:
            self._changed(spacing=True)

    def _keep(self):
        R = self.R.value
        if self.shape.value == "custom mask":
            if self.mask is None:
                return kf.shape_keep("circle", R)
            return kf.mask_keep(self.mask, R, invert=self.invert.value)
        return kf.shape_keep(self.shape.value, R)

    def _points(self):
        """Dots of the current outline (fast: no gates, no tracing)."""
        half = self.sp["native"] / 2 - 1; du, dv, keep = self.du.value, self.dv.value, self._keep()
        ok = lambda u, v: keep(u, v) and abs(u * du) < half and abs(v * dv) < half
        g = kf.build_grid_shape(4 * int(np.ceil(1.5 * self.R.value)) + 1, ok)
        return g

    def _changed(self, spacing=False):
        self._dirty = True
        if self.sp is None:
            return
        if spacing:
            self._draw_spectrum(); self._draw_mask()

    def _colours(self):
        return dict(bg=self.c_bg.value, line=self.c_line.value, lv=[c.value for c in self.c_lv],
                    g4=kf.PALETTES[self.palette.value]["g4"], g3=kf.PALETTES[self.palette.value]["g3"],
                    dot=self.c_dot.value)

    def _apply_palette(self):
        P = kf.PALETTES[self.palette.value]
        self.c_bg.value, self.c_line.value = P["bg"], P["line"]
        for c, v in zip(self.c_lv, P["lv"]):
            c.value = v

    def build(self):
        """Build (or rebuild) the kolam from the current settings."""
        self.K = kf.build_kolam(self.sp, self.du.value, self.dv.value, self._keep(), self.R.value,
                                lattice=self.lat, method=self.method.value, seed=self.seed.value)
        self._dirty = False
        return self.K

    def render(self, width_px):
        if self._dirty or self.K is None:
            self.build()
        return kf.render_kolam(self.K, self._colours(), width_px=width_px, dots=self.pulli.value,
                               petals=self.petals.value, glow=self.glow.value,
                               transparent=self.transparent.value, dot_size=self.dot_size.value,
                               tip=self.tip.value)

    # ------------------------------------------------------------------ drawing
    def _draw_image(self):
        with self.img_out:
            clear_output(wait=True)
            fig, ax = plt.subplots(1, 2, figsize=(11, 5))
            ax[0].imshow(self.img, cmap="gray"); ax[0].axis("off"); ax[0].set_title("input (square, greyscale)")
            L = np.log10(self.sp["amp"] + 1e-12); c, u = self.sp["cc"], self.sp["up"]
            z = int(160 * u); crop = L[c - z:c + z:4, c - z:c + z:4]
            ax[1].imshow(crop, cmap="inferno", origin="lower", extent=[-160, 160, -160, 160],
                         vmin=np.percentile(crop, 40), vmax=crop.max())
            ax[1].set_title("Fourier transform (log amplitude)"); ax[1].set_xlabel("u"); ax[1].set_ylabel("v")
            plt.show()
            n = self.sp["native"]
            print(f"lattice: a* = {self.lat['a_star_native']:.2f}, b* = {self.lat['b_star_native']:.2f} cycles per image"
                  f"  (pitch {n/self.lat['a_star_native']:.1f} × {n/self.lat['b_star_native']:.1f} px),"
                  f"  contrast {self.lat['contrast']:.2f}")

    def _draw_spectrum(self):
        g = self._points()
        pts = [g["uv"][d] for d in g["dots"]]
        S = kf.sample_spacing(self.sp, self.du.value, self.dv.value, pts, self.lat)
        with self.spec_out:
            clear_output(wait=True)
            xy = np.array([(p[0] * self.du.value, p[1] * self.dv.value) for p in S["pts"]])
            on = np.array([xy[i] for i, p in enumerate(S["pts"]) if S["on_lattice"][p]]).reshape(-1, 2)
            reach = min(np.abs(xy).max() * 1.12 + 3, self.sp["native"] / 2)
            L = np.log10(self.sp["amp"] + 1e-12); c, u = self.sp["cc"], self.sp["up"]
            z = int(reach * u); step = max(1, int(2 * z / 800))
            crop = L[c - z:c + z:step, c - z:c + z:step]
            fig, ax = plt.subplots(figsize=(7.5, 7.5))
            ax.imshow(crop, cmap="inferno", origin="lower", extent=[-reach, reach, -reach, reach],
                      vmin=np.percentile(crop, 40), vmax=crop.max())
            ax.scatter(xy[:, 0], xy[:, 1], s=max(2, 3000 / len(xy)), c="#66ffe0", alpha=.8, linewidths=0)
            if len(on):
                ax.scatter(on[:, 0], on[:, 1], s=max(20, 12000 / len(xy)), facecolors="none",
                           edgecolors="#66ffe0", linewidths=1.2)
            ax.set_xlabel("u (cycles across)"); ax.set_ylabel("v (cycles up)")
            ax.set_title(f"{len(S['pts'])} sample points, {len(on)} on diffraction orders (rings)")
            plt.show()

    def _draw_mask(self):
        g = self._points()
        pts = [g["uv"][d] for d in g["dots"]]
        S = kf.make_levels_spacing(self.sp, self.du.value, self.dv.value, pts, self.lat)
        with self.mask_out:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            X = [kf.to_draw(d)[0] for d in g["dots"]]; Y = [kf.to_draw(d)[1] for d in g["dots"]]
            ax.scatter(X, Y, c=[kf.LEVEL_COLOURS[S["level"][g["uv"][d]]] for d in g["dots"]],
                       s=max(3, 9000 / len(X)), marker="s")
            ax.set_aspect("equal"); ax.axis("off")
            sym = kf.is_symmetric(g)
            ax.set_title(f"{len(g['dots'])} dots · {'symmetric' if sym else 'not symmetric (random choices)'}")
            plt.show()

    def _render_preview(self):
        with self.style_out:
            clear_output(wait=True)
            print("building and rendering …")
            im = self.render(1000)
            clear_output(wait=True)
            if im.mode == "RGBA":                                  # show transparency on a checkerboard
                t = 20; cb = (np.indices(im.size[::-1]).sum(0) // t % 2) * 40 + 200
                bg = Image.fromarray(np.dstack([cb] * 3).astype(np.uint8)).convert("RGBA")
                bg.alpha_composite(im); show = bg.convert("RGB")
            else:
                show = im
            display(show.resize((720, 720)))
            s = self.K["stats"]
            print(f"{s['dots']} dots · {s['loops']} loops · longest {100*s['longest']:.0f}% · "
                  f"{s['wrong']} dots off their level · {s['orders_bright']}/{s['orders']} diffraction orders "
                  f"drawn as fan or diamond · gate rule: {s['method']}" + ("" if s["symmetric"] else " (not symmetric)"))
            self.preview = im

    def _export(self):
        with self.export_out:
            clear_output(wait=True)
            px = self.width_mm.value / 25.4 * self.dpi.value
            if px > 14000:
                print(f"{px:.0f} px is very large; lower the dpi or width."); return
            print(f"rendering at {px:.0f} px …")
            im = self.render(int(px * 1.12))
            os.makedirs(self.folder.value, exist_ok=True)
            path = os.path.join(self.folder.value, f"{self.fname.value}.{self.fmt.value}")
            path, note = kf.export_image(im, path, self.fmt.value, self.width_mm.value, self.dpi.value,
                                         flatten_bg=self.c_bg.value)
            clear_output(wait=True)
            print("saved:", note); display(FileLink(path))

    # ------------------------------------------------------------------ panels
    def image_panel(self):
        return W.VBox([W.HBox([self.path, self.load_btn]), self.upload, self.img_out])

    def spacing_panel(self):
        return W.VBox([W.HTML("<b>Sample spacing in frequency space</b> (cycles per image). "
                              "The buttons put samples on every order, or 2–4 samples per order."),
                       W.HBox(self.div_btns), self.du, self.dv, self.R, self.spec_out])

    def mask_panel(self):
        return W.VBox([W.HTML("<b>Outline.</b> Pick a shape, or upload a mask image (dark = inside; "
                              "it is stretched over the outline size)."),
                       self.shape, W.HBox([self.mask_up, self.invert]), self.mask_out])

    def style_panel(self):
        colours = W.VBox([self.palette, W.HBox([self.c_bg, self.c_line]),
                          W.HBox(self.c_lv[:2]), W.HBox(self.c_lv[2:]), self.c_dot])
        flags = W.VBox([self.transparent, self.pulli, self.petals, self.glow, self.dot_size, self.tip,
                        self.method, self.seed])
        return W.VBox([W.HBox([colours, flags]), self.render_btn, self.style_out])

    def export_panel(self):
        return W.VBox([self.fmt, self.width_mm, self.dpi, self.folder, self.fname, self.export_btn,
                       self.export_out])
