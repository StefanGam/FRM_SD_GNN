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


def make_animation_artist(network_list, pos, output_path: Path, interval: int = 700, transparent: bool = False):
    """Use ArtistAnimation for complete frame control - each frame is completely independent"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up the figure properly for transparency
    if transparent:
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0)
        ax.patch.set_facecolor('none')
        ax.patch.set_alpha(0)
    
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Pre-create all frames as independent artist collections
    print("Pre-rendering all frames...")
    frames = []
    
    for frame_idx, (date, G) in enumerate(network_list):
        # Clear for this frame
        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')
        
        if transparent:
            ax.patch.set_facecolor('none')
            ax.patch.set_alpha(0)
        
        # Filter edges for this frame
        edges_to_draw = []
        edge_weights = []
        
        for u, v, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            if weight > 0.5:
                edges_to_draw.append((u, v))
                edge_weights.append(weight)
        
        G_filtered = G.edge_subgraph(edges_to_draw).copy()
        
        frame_artists = []
        
        # Draw nodes
        node_collection = nx.draw_networkx_nodes(
            G, pos, ax=ax, 
            node_color='lightblue', 
            node_size=1200, 
            edgecolors='darkblue', 
            linewidths=2,
            alpha=0.9
        )
        frame_artists.append(node_collection)
        
        # Draw edges
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
            
            edge_collection = nx.draw_networkx_edges(
                G_filtered, pos, ax=ax, 
                arrowstyle='-|>', 
                arrowsize=15, 
                edge_color='gray', 
                width=widths,
                alpha=0.7,
                connectionstyle="arc3,rad=0.1"
            )
            if edge_collection:
                frame_artists.append(edge_collection)
        
        # Draw labels
        labels = nx.draw_networkx_labels(
            G, pos, ax=ax, 
            font_size=8, 
            font_weight='bold',
            font_color='black'
        )
        frame_artists.extend(labels.values())
        
        # Add title
        title = ax.set_title(
            f"SD Network - {date.strftime('%Y-%m-%d')}\n"
            f"{G.number_of_nodes()} nodes, {G_filtered.number_of_edges()}/{G.number_of_edges()} significant edges", 
            fontsize=14, fontweight='bold'
        )
        frame_artists.append(title)
        
        # Add frame counter
        frame_text = ax.text(
            0.02, 0.98, 
            f"Frame {frame_idx+1}/{len(network_list)}", 
            transform=ax.transAxes, 
            fontsize=10, 
            ha='left', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        frame_artists.append(frame_text)
        
        frames.append(frame_artists)
        
        if frame_idx % 5 == 0:
            print(f"  Rendered frame {frame_idx+1}/{len(network_list)}")
    
    print("Creating animation...")
    
    # Create ArtistAnimation - each frame completely replaces the previous
    ani = animation.ArtistAnimation(
        fig, frames, interval=interval, repeat=True, blit=False
    )

    print(f"Saving SD network animation to {output_path} (transparent={transparent}) ...")
    
    save_kwargs = {
        'writer': 'pillow',
        'fps': 1000 // interval,
    }
    if transparent:
        save_kwargs['savefig_kwargs'] = {
            'transparent': True, 
            'facecolor': 'none'
        }
    else:
        save_kwargs['savefig_kwargs'] = {
            'facecolor': 'white'
        }

    ani.save(str(output_path), **save_kwargs)
    plt.close(fig)
    print("Done! Animation saved.")


def main():
    parser = argparse.ArgumentParser(description="Animate SD networks using ArtistAnimation (no frame piling)")
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
    make_animation_artist(network_list, pos, output_path, interval=args.interval, transparent=args.transparent)


if __name__ == '__main__':
    main()