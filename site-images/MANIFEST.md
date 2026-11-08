# Site images

For the portfolio project page. All are 1600 × 1000 (16:10) WebP on a dark
background, with no metadata. Site photos are cropped to the ceiling: no
people, signage, logos or drawings.

| File | Caption | Alt text | What it shows |
| --- | --- | --- | --- |
| `01-pipe-vs-paint.webp` | A ceiling view cut from a 360° photo, and every candidate line labelled by a vision-language judge: pipe, paint or cable. | Top: a bare concrete ceiling with red pipes and red painted stripes. Bottom: the same view darkened, with lines drawn over it in green for pipe, magenta for paint and blue for cable. | The core result of the method: thin lines from the tube filter, each named by Claude Haiku. Lead image. |
| `02-sam3-mask-vs-tube-lines.webp` | Asked for "red pipe", SAM 3 returns one mask over slab, paint and pipes. A tube filter on redness draws thin lines a judge can label. | Top: the ceiling view with a blue mask covering most of the slab and the painted stripes. Bottom: the same view with thin white lines along each red pipe and stripe. | The failure that motivated the tube filter and the judge. |
| `03-panorama-to-flat-views.webp` | One 360° photo cut into 16 ordinary flat views, the input every detector sees. | A wide cafeteria panorama with coloured outlines of 16 camera footprints, above a strip of the 16 flat views. | The projection step, on a free CC0 panorama from Poly Haven (not the site). |
| `04-pipeline.webp` | From a 360° photo to the fire pipe on the plan: capture, flat views, candidates, judge, plan, network. | A six-step diagram in coloured boxes: capture, cut into flat views, find candidates, judge each line, place on the plan, pipe network. | The whole pipeline at a glance, with the open problem (pipe height) noted at the bottom. |

Larger, uncropped versions of the same figures are in `docs/figures/` for the
README.
