# animate_network_webm_alpha.py
import shutil
from pathlib import Path
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import networkx as nx

# ---------- Config ----------
NETWORKS_PATH   = Path("outputs/networks.pkl")                  # [(date, G), ...]
OUTPUT_WEBM     = Path("outputs/sd_network_evolution.webm")     # transparent video
FRAME_INTERVAL  = 700  # ms per frame
FPS             = 1000 / FRAME_INTERVAL

# If ffmpeg isn't on PATH, set your explicit path here:
# Example for portable install:
# FFMPEG_EXPLICIT = r"C:\ffmpeg\bin\ffmpeg.exe"
FFMPEG_EXPLICIT = None
# --------------------------------

# ---- Find ffmpeg ----
ffmpeg_path = shutil.which("ffmpeg") or FFMPEG_EXPLICIT or \
              next((p for p in [
                  r"C:\Program Files\FFmpeg\bin\ffmpeg.exe",
                  r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                  r"C:\ffmpeg\bin\ffmpeg.exe",
              ] if Path(p).exists()), None)

if not ffmpeg_path:
    raise FileNotFoundError(
        "FFmpeg not found.\n"
        "Install with `winget install --id=Gyan.FFmpeg -e` and restart PyCharm,\n"
        "or download a zip to C:\\ffmpeg and set FFMPEG_EXPLICIT = r\"C:\\ffmpeg\\bin\\ffmpeg.exe\"."
    )

mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
print(f"Using FFmpeg at: {ffmpeg_path}")

# ---- Load precomputed networks ----
with open(NETWORKS_PATH, "rb") as f:
    network_list = pickle.load(f)  # list of tuples: [(datetime, nx.DiGraph), ...]

# ---- Fixed layout over union graph ----
all_nodes = sorted({n for _, G in network_list for n in G.nodes})
G_union = nx.DiGraph()
G_union.add_nodes_from(all_nodes)
for _, G in network_list:
    G_union.add_edges_from(G.edges())

# For reproducibility you can switch to circular:
# pos = nx.circular_layout(G_union)
pos = nx.spring_layout(G_union, seed=42)

# ---- Figure (fully transparent) ----
fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
fig.set_facecolor("none"); fig.patch.set_alpha(0.0)
ax.set_facecolor("none");  ax.patch.set_alpha(0.0)

def update(frame):
    ax.clear()
    # keep transparency after clear
    ax.set_facecolor("none"); ax.patch.set_alpha(0.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axis("off")

    date, G = network_list[frame]

    # --- Draw network (customize as needed) ---
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color="skyblue", node_size=800,
        edgecolors="k", linewidths=1
    )
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        arrowstyle="-|>", arrowsize=15,
        edge_color="gray", width=2
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=13)

    ax.set_title(f"SD Network - {date.strftime('%Y-%m-%d')}", fontsize=18)

ani = animation.FuncAnimation(
    fig, update, frames=len(network_list),
    interval=FRAME_INTERVAL, repeat=True, blit=False
)

# ---- Save as WebM with alpha (no piling up) ----
OUTPUT_WEBM.parent.mkdir(parents=True, exist_ok=True)

writer = FFMpegWriter(
    fps=FPS,
    codec="libvpx-vp9",
    # yuva420p keeps the alpha channel; adjust bitrate as desired
    extra_args=["-pix_fmt", "yuva420p", "-b:v", "4M"]
)

print(f"Saving to {OUTPUT_WEBM} ...")
ani.save(
    str(OUTPUT_WEBM),
    writer=writer,
    dpi=150,
    savefig_kwargs={"facecolor": "none", "transparent": True}
)
print("Done! Transparent WebM saved.")
