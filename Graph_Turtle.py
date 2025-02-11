from Pre_Turtle import *
from Pre_Turtle_2 import *
import turtle
import networkx as nx

# Import the skeleton image from the previous step
Image = skeleton_cleaned  # For gradient operator - change it to skeleton

def skeleton_to_graph(skel):
    """
    Converts a skeleton image (0/1) to a pixel-level undirected graph using 8-connectivity.
    Each '1' pixel is a node; edges connect neighboring '1' pixels.
    """
    G = nx.Graph()
    rows, cols = skel.shape

    for r in range(rows):
        for c in range(cols):
            if skel[r, c] == 1:
                G.add_node((r, c))
                # Check neighbors (8-connectivity)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                               (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < rows and 0 <= cc < cols:
                        if skel[rr, cc] == 1:
                            G.add_edge((r, c), (rr, cc))
    return G

def find_branches_in_skeleton(G):
    """
    Extracts a list of vessel 'branches' from a skeleton graph G.

    Each branch is a path between two 'break' nodes:
      - Endpoints (degree == 1)
      - Junctions (degree >= 3)
    Nodes with degree == 2 are considered 'intermediate' and
    lie along the branch rather than splitting it.

    Returns: list of paths, each path = [ (r1, c1), (r2, c2), ... ]
    """
    all_branches = []
    # Process each connected component separately.
    for comp in nx.connected_components(G):
        subG = G.subgraph(comp).copy()
        # Identify break nodes (endpoints or junctions).
        break_nodes = set(n for n in subG.nodes if subG.degree(n) != 2)

        # If there are no break nodes, the component is a cycle or a chain.
        if not break_nodes:
            # For a cycle (or pure chain), simply get one traversal.
            some_node = next(iter(subG.nodes))
            branch = list(nx.dfs_tree(subG, source=some_node).nodes())
            all_branches.append(branch)
            continue

        # To avoid extracting the same branch twice, track visited edges.
        visited_edges = set()

        # For each break node, walk along each adjacent branch.
        for bn in break_nodes:
            for nb in subG.neighbors(bn):
                edge = tuple(sorted((bn, nb)))
                if edge in visited_edges:
                    continue
                # Start a new branch from bn to nb.
                branch = [bn, nb]
                visited_edges.add(edge)
                prev = bn
                current = nb

                # Follow the chain until reaching a break node.
                # For degree-2 nodes, there should be exactly one neighbor that is not the previous node.
                while current not in break_nodes:
                    neighbors = list(subG.neighbors(current))
                    # Since current has degree 2, one neighbor is 'prev', the other is next.
                    next_node = neighbors[0] if neighbors[0] != prev else neighbors[1]
                    branch.append(next_node)
                    # Mark the traversed edge as visited.
                    visited_edges.add(tuple(sorted((current, next_node))))
                    prev, current = current, next_node
                all_branches.append(branch)

    return all_branches

def display_results(skel, G, all_branches):
    """
    Displays the skeleton image, the graph representation, and the traced branches.
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Skeleton Image
    axes[0].imshow(skel, cmap='gray')
    axes[0].set_title("Skeleton Image")
    axes[0].axis("off")

    # Graph Representation
    pos = {node: (node[1], -node[0]) for node in G.nodes()}  # Flip y-axis for correct orientation
    nx.draw(G, pos, node_size=1, edge_color='cyan', linewidths=0.5, ax=axes[1])
    axes[1].set_title("Skeleton to Graph")
    axes[1].set_aspect('equal')  # Keep aspect ratio consistent
    axes[1].set_xlim([0, skel.shape[1]])  # Ensure proper scaling
    axes[1].set_ylim([-skel.shape[0], 0])  # Flip y-axis to align correctly

    # Traced Branches
    colors = ['r', 'g', 'b', 'm', 'y', 'c']
    for i, branch in enumerate(all_branches):
        x = [node[1] for node in branch]
        y = [-node[0] for node in branch]  # Flip y for proper orientation
        axes[2].plot(x, y, marker='o', markersize=1, linestyle='-', linewidth=0.5, color=colors[i % len(colors)])
    axes[2].set_title("Extracted Branches")
    axes[2].set_aspect('equal')  # Maintain aspect ratio
    axes[2].set_xlim([0, skel.shape[1]])
    axes[2].set_ylim([-skel.shape[0], 0])
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()
    return fig


def draw_branches_with_turtle(branches, image_shape, scale=1.0):
    """
    Draw each branch using Python turtle.

    Args:
        branches (list): List of paths. Each path is a list of (row, col) pixel coords.
        image_shape (tuple): (height, width) of the skeleton image.
        scale (float): Scale factor for drawing.
    """
    h, w = image_shape

    # Set up turtle screen to match image dimensions (scaled)
    screen = turtle.Screen()
    screen.setup(width=w * scale, height=h * scale)
    # Center the coordinate system
    screen.setworldcoordinates(-w / 2 * scale, -h / 2 * scale, w / 2 * scale, h / 2 * scale)

    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    t.penup()

    # Disable animation for faster drawing
    turtle.tracer(0, 0)

    for path in branches:
        if not path:
            continue
        # Convert the first pixel coordinate from image (row, col) to turtle (x, y)
        first_row, first_col = path[0]
        x = (first_col - w / 2) * scale
        y = (h / 2 - first_row) * scale
        t.penup()
        t.goto(x, y)
        t.pendown()
        for (row, col) in path[1:]:
            x = (col - w / 2) * scale
            y = (h / 2 - row) * scale
            t.goto(x, y)

    turtle.update()
    turtle.done()


# Example usage:
if __name__ == "__main__":
    image_shape = Image.shape

    # Step 1: Convert the skeleton to a graph
    G = skeleton_to_graph(Image)

    # Step 2: Find meaningful vessel branches
    branches = find_branches_in_skeleton(G)

    # Step 3: Display the results
    fig = display_results(Image, G, branches)

    # Step 4: Draw branches using the corrected coordinate transform
    draw_branches_with_turtle(branches, image_shape, scale=1.0)
