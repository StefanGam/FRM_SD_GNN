import argparse
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path


def load_networks(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_union_layout(network_list, seed=42):
    all_nodes = set()
    for _, G in network_list:
        all_nodes.update(G.nodes)
    all_nodes = sorted(all_nodes)
    G_union = nx.DiGraph()
    G_union.add_nodes_from(all_nodes)
    for _, G in network_list:
        G_union.add_edges_from(G.edges())
    pos = nx.spring_layout(G_union, seed=seed)
    return pos


def make_animation(network_list, pos, output_path: Path, interval: int = 700, transparent: bool = False):
    plt.ioff()  # Turn off interactive plotting
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    def update(frame):
        # Completely clear and reset the axes
        ax.clear()
        
        # Set fixed axis properties every frame
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # ALWAYS use white background during drawing (even for transparent output)
        fig.patch.set_facecolor('white')
        fig.patch.set_alpha(1)
        ax.patch.set_facecolor('white')
        ax.patch.set_alpha(1)
        
        date, G = network_list[frame]
        
        # Filter edges for visibility
        edges_to_draw = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            if weight > 0.5:  # Only show significant edges
                edges_to_draw.append((u, v))
                edge_weights.append(weight)
        
        G_filtered = G.edge_subgraph(edges_to_draw).copy()
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax, 
            node_color='lightblue', 
            node_size=1200, 
            edgecolors='darkblue', 
            linewidths=2,
            alpha=0.9
        )
        
        # Draw filtered edges
        if G_filtered.number_of_edges() > 0:
            if edge_weights:
                max_weight = max(edge_weights)
                min_weight = min(edge_weights)
                if max_weight > min_weight:
                    widths = [1 + 4 * (w - min_weight) / (max_weight - min_weight) for w in edge_weights]
                else:
                    widths = [2.0] * len(edge_weights)
            else:
                widths = [2.0] * G_filtered.number_of_edges()
            
            nx.draw_networkx_edges(
                G_filtered, pos, ax=ax, 
                arrowstyle='-|>', 
                arrowsize=15, 
                edge_color='gray', 
                width=widths,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1"
            )
        
        # Draw labels
        nx.draw_networkx_labels(
            G, pos, ax=ax, 
            font_size=8, 
            font_weight='bold',
            font_color='black'
        )
        
        # Add title
        ax.set_title(
            f"SD Network - {date.strftime('%Y-%m-%d')}\n"
            f"{G.number_of_nodes()} nodes, {G_filtered.number_of_edges()}/{G.number_of_edges()} significant edges", 
            fontsize=14, fontweight='bold',
            color='black'
        )
        
        # Add frame counter
        ax.text(
            0.02, 0.98, 
            f"Frame {frame+1}/{len(network_list)}", 
            transform=ax.transAxes, 
            fontsize=10, 
            ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            color='black'
        )

    # Create animation - ALWAYS use opaque rendering during animation creation
    ani = animation.FuncAnimation(
        fig, update, frames=len(network_list), interval=interval, 
        repeat=True, blit=False, cache_frame_data=False
    )

    print(f"Saving SD network animation to {output_path} (transparent={transparent}) ...")
    
    # Create writer - handle transparency ONLY at save time, not during frame rendering
    writer = animation.PillowWriter(fps=1000//interval)
    
    if transparent:
        # For transparent: save with transparent background but all frames were rendered opaque
        ani.save(str(output_path), writer=writer, savefig_kwargs={'transparent': True, 'facecolor': 'none'})
    else:
        # For opaque: save with white background
        ani.save(str(output_path), writer=writer, savefig_kwargs={'facecolor': 'white'})
    
    plt.close(fig)  # Close figure to free memory
    print("Done! Animation saved.")


def main():
    parser = argparse.ArgumentParser(description="Animate SD networks from a pickle of networks")
    parser.add_argument('networks_path', nargs='?', default='outputs/networks.pkl', help='Path to networks.pkl')
    parser.add_argument('--output', '-o', default=None, help='Output GIF path (default: same folder as networks file)')
    parser.add_argument('--interval', '-i', type=int, default=700, help='Milliseconds per frame')
    parser.add_argument('--transparent', '-t', action='store_true', help='Save GIF with transparent background')
    parser.add_argument('--last', '-l', type=int, default=None, help='If set, only animate the last N frames (e.g., 90)')
    parser.add_argument('--skip', '-s', type=int, default=1, help='Skip every N frames to make changes more visible (default: 1, no skipping)')
    args = parser.parse_args()

    networks_path = Path(args.networks_path)
    if not networks_path.exists():
        raise FileNotFoundError(f"Networks file not found: {networks_path}")

    network_list = load_networks(networks_path)
    if args.last is not None:
        if args.last <= 0:
            raise ValueError("--last must be > 0")
        network_list = network_list[-args.last:]
    
    # Apply frame skipping if requested
    if args.skip > 1:
        network_list = network_list[::args.skip]
        print(f"Skipping every {args.skip} frames, showing {len(network_list)} frames total")
    pos = build_union_layout(network_list)

    output_path = Path(args.output) if args.output else (networks_path.parent / 'sd_network_evolution.gif')
    make_animation(network_list, pos, output_path, interval=args.interval, transparent=args.transparent)


if __name__ == '__main__':
    main()
