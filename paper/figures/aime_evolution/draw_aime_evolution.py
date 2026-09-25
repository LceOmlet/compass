"""Render the AIME candidate tree from the verified figure data.

Run with Python and Matplotlib. The layout follows the annotated-tree
presentation in GEPA Figure 5, https://arxiv.org/pdf/2507.19457#page=7.
Only the final selected candidate has a held-out test score in this figure.
"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent

report = json.loads((ROOT / 'aime_evolution_data.json').read_text(encoding='utf-8'))
parents = report['parents']
assert parents == [None, 0, 1, 1, 2, 3, 3, 0, 5, 8, 3, 9, 6, 9, 9, 14]
chain = report['chain']
assert chain == [0, 1, 3, 5, 8, 9, 11]
assert all(parents[b] == a for a, b in zip(chain, chain[1:]))
assert len(parents) == 16

positions = {
    0: (4.43, 5.10),
    1: (4.43, 4.59), 7: (6.17, 4.59),
    3: (4.43, 3.93), 2: (6.17, 3.93),
    5: (4.43, 3.27), 6: (5.03, 3.27), 10: (5.59, 3.27), 4: (6.17, 3.27),
    8: (4.43, 2.61), 12: (5.03, 2.61),
    9: (4.43, 1.95),
    11: (4.43, 1.29), 13: (5.22, 1.29), 14: (6.01, 1.29),
    15: (6.01, .60),
}
assert set(positions) == set(range(16))

plt.rcParams.update({
    'font.family': 'Times New Roman', 'font.size': 10,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
})
W, YMIN, YMAX = 6.6, .34, 5.31
H = YMAX - YMIN
fig = plt.figure(figsize=(W, H), facecolor='white')
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(YMIN, YMAX)
ax.axis('off')

INK = '#252525'
GRAY = '#C8CDD0'
EDGE = '#555B60'
RED = '#B14436'
FINAL = '#D6E8E8'
boxes = []
texts = []

def text(x, y, value, size=10, weight='normal', color=INK,
         ha='center', va='center', box=None):
    artist = ax.text(x, y, value, fontsize=size, fontweight=weight,
                     color=color, ha=ha, va=va, linespacing=1.13, zorder=6)
    texts.append((artist, box))
    return artist

text(4.08, 5.10, 'Seed', size=10, color=EDGE, ha='right')

# The full recorded tree. Only the selected ancestry uses red arrows.
radius = .158
path_edges = set(zip(chain, chain[1:]))
for child, parent in enumerate(parents):
    if parent is None:
        continue
    x1, y1 = positions[parent]
    x2, y2 = positions[child]
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** .5
    inset = radius + .012
    start = (x1 + dx / length * inset, y1 + dy / length * inset)
    end = (x2 - dx / length * inset, y2 - dy / length * inset)
    on_path = (parent, child) in path_edges
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                                mutation_scale=7, linewidth=1.20 if on_path else .70,
                                color=RED if on_path else '#92989D', zorder=1,
                                shrinkA=0, shrinkB=0))

for candidate, (x, y) in positions.items():
    on_path = candidate in chain
    selected = candidate == report['selected_candidate_idx']
    ax.add_patch(Circle((x, y), radius,
                        facecolor=FINAL if selected else ('white' if on_path else '#E1E3E5'),
                        edgecolor=INK if on_path else '#8E959A',
                        linewidth=1.1 if on_path else .7, zorder=3))
    text(x, y, str(candidate), size=10.5,
         weight='bold' if selected else 'normal', color=INK if on_path else '#62686D')

annotations = {
    1: ('Add a solving checklist',
        'Check constraints, split into cases,\nand verify edge cases and the final answer.'),
    3: ('Specify the answer format',
        'Require a plain integer when requested,\nwithout additional text or formatting.'),
    5: ('Revise the solving checklist',
        'Check bounds after substitution.\nRemove the explicit integer-output instruction.'),
    8: ('Check domains and justify assumptions',
        'Add explicit domain checks for substitutions\nand require assumptions to be justified.'),
    9: ('Expand the output instructions',
        'Require a single integer. Add two conversion\nexamples and a list of incorrect output formats.'),
    11: ('Retain the rule and remove examples',
         'Keep the integer-output rule. Remove those two\nconversion examples and the incorrect-output list.'),
}
left, right, height = .20, 4.07, .545
for candidate, (heading, body) in annotations.items():
    node_x, y = positions[candidate]
    rect = (left, y - height / 2, right, y + height / 2)
    boxes.append(rect)
    ax.add_patch(FancyBboxPatch((left, y - height / 2), right - left, height,
                               boxstyle='round,pad=0.012,rounding_size=0.055',
                               facecolor='#FAFAFA', edgecolor='#5B5B5B', linewidth=.75,
                               zorder=4))
    ax.plot([right + .02, node_x - radius - .04], [y, y],
            color='#A4AAAE', linewidth=.6, linestyle=(0, (1.5, 2)), zorder=2)
    text((left + right) / 2, y + .152, heading, size=10.3, weight='bold', box=rect)
    text((left + right) / 2, y - .077, body, size=9.8, box=rect)

assert set(annotations) == set(chain) - {0}
for first, second in zip(boxes, boxes[1:]):
    assert first[1] - second[3] > .075

text(4.43, .90, 'Selected', size=9.8, weight='bold')
text(4.43, .67, f"{report['final_test_score']:.2f}% test", size=10.5)

# Check text placement and ensure the summaries stay inside their boxes.
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for artist, box in texts:
    bb = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
    assert bb.x0 > .02 and bb.x1 < W - .02 and bb.y0 > YMIN + .02 and bb.y1 < YMAX - .02, (artist.get_text(), bb)
    if box:
        x0, y0, x1, y1 = box
        assert x0 + .04 <= bb.x0 and bb.x1 <= x1 - .04, (artist.get_text(), bb)
        assert y0 + .025 <= bb.y0 and bb.y1 <= y1 - .025, (artist.get_text(), bb)

fig.savefig(ROOT / 'aime_evolution.pdf', metadata={'Title': 'AIME program evolution', 'Author': ''})
fig.savefig(ROOT / 'aime_evolution.svg')
fig.savefig(ROOT / 'aime_evolution.png', dpi=280)
print('Rendered all 16 recorded candidates and 15 parent edges. Highlighted the 6 edges to candidate 11.')
