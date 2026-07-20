"""Generate the EPUB cover image (assets/epub-cover.png), 1600x2560.

Deterministic, dependency-light (matplotlib only), matching the book's
palette: dark slate ground, teal accent rule, typographic title block.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

W, H = 1600, 2560
fig = plt.figure(figsize=(W / 200, H / 200), dpi=200)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

ax.add_patch(Rectangle((0, 0), W, H, color="#1d2b36"))
ax.add_patch(Rectangle((0, H - 340), W, 12, color="#147d92"))
ax.add_patch(Rectangle((0, 320), W, 6, color="#31708a"))

# Node-and-edge motif: a restrained diagonal of linked squares.
for i, (x, y) in enumerate([(260, 820), (560, 1010), (900, 900), (1220, 1080)]):
    ax.add_patch(Rectangle((x - 34, y - 34), 68, 68, fill=False,
                           edgecolor="#2e5a6e", linewidth=2.5))
    if i:
        px, py = [(260, 820), (560, 1010), (900, 900), (1220, 1080)][i - 1]
        ax.plot([px + 34, x - 34], [py, y], color="#2e5a6e", linewidth=2.0)

ax.text(W / 2, 1840, "Agentic", ha="center", va="center",
        fontsize=52, fontweight="bold", color="#e9eef2", family="DejaVu Sans")
ax.text(W / 2, 1620, "Engineering", ha="center", va="center",
        fontsize=52, fontweight="bold", color="#e9eef2", family="DejaVu Sans")
ax.text(W / 2, 1400, "From neural networks to dependable agent systems",
        ha="center", va="center", fontsize=15.5, color="#9fb3c0",
        family="DejaVu Sans")
ax.text(W / 2, 480, "Omkar Sarde", ha="center", va="center",
        fontsize=22, color="#cfd9df", family="DejaVu Sans")

fig.savefig("assets/epub-cover.png", dpi=200)
print("wrote assets/epub-cover.png")
