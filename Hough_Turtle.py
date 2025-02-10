from Pre_Turtle import *
import turtle
import networkx as nx

# Import the skeleton image from the previous step
Image = skeleton_cleaned

# Horizontal
kernel_h = np.array([[-1,  2, -1],
                     [-1,  2, -1],
                     [-1,  2, -1]], dtype=np.float32)

# Vertical
kernel_v = np.array([[-1, -1, -1],
                     [ 2,  2,  2],
                     [-1, -1, -1]], dtype=np.float32)

# +45 degree
kernel_45p = np.array([[ 2, -1, -1],
                       [-1,  2, -1],
                       [-1, -1,  2]], dtype=np.float32)

# -45 degree
kernel_45m = np.array([[-1, -1,  2],
                       [-1,  2, -1],
                       [ 2, -1, -1]], dtype=np.float32)

def line_detection_convolution(img, kernels, threshold=1):
    # We'll accumulate maximum response across all kernels
    h, w = img.shape
    response_stack = np.zeros((h, w, len(kernels)), dtype=np.float32)

    for i, K in enumerate(kernels):
        # Filter using signal's convolve2d
        resp = signal.convolve2d(img, K, boundary='symm', mode='same')
        response_stack[..., i] = resp

    # Take maximum response across all kernels
    max_resp = np.max(response_stack, axis=2)

    # Threshold
    line_mask = (max_resp > threshold).astype(np.float32)
    return line_mask

def track_lines_in_mask(line_mask):
    visited = np.zeros_like(line_mask, dtype=bool)
    h, w = line_mask.shape
    paths = []

    def neighbors(r, c):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc
    # What is DFS?
    # Depth-first search (DFS) is an algorithm for traversing or searching tree or graph data structures. It starts at the root node and explores as far as possible along each branch before backtracking.
    def dfs(r, c): # Depth-first search -
        stack = [(r, c)]
        path_pixels = []
        while stack:
            rr, cc = stack.pop()
            if not visited[rr, cc]:
                visited[rr, cc] = True
                path_pixels.append((rr, cc))
                # push neighbors that are line pixels
                for nr, nc in neighbors(rr, cc):
                    if line_mask[nr, nc] == 1 and not visited[nr, nc]:
                        stack.append((nr, nc))
        return path_pixels

    for row in range(h):
        for col in range(w):
            if line_mask[row, col] == 1 and not visited[row, col]:
                path = dfs(row, col)
                if len(path) > 500:
                    paths.append(path)
    return paths

def draw_paths(paths, image_shape, scale=1):
    """
    Draws a set of paths using turtle graphics.

    Args:
        paths (list): List of paths. Each path is a list of (row, col) tuples.
        image_shape (tuple): The (height, width) of the original image.
        scale (float): Factor to scale the drawing. Increase if your image is small.
    """
    h, w = image_shape

    # Set up turtle screen to match image dimensions (scaled)
    screen = turtle.Screen()
    screen.setup(width=w * scale, height=h * scale)
    # Optionally, set the coordinate system so that (0,0) is at the center
    screen.setworldcoordinates(-w / 2 * scale, -h / 2 * scale, w / 2 * scale, h / 2 * scale)

    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    t.penup()

    # Disable animation for faster drawing
    turtle.tracer(0, 0)

    # Iterate over each path
    for path in paths:
        if not path:
            continue
        # Transform the first pixel from image coordinates (row, col) to turtle coordinates (x, y)
        first_row, first_col = path[0]
        x = (first_col - w / 2) * scale
        y = (h / 2 - first_row) * scale
        t.penup()
        t.goto(x, y)
        t.pendown()
        # Draw the rest of the path
        for (row, col) in path[1:]:
            x = (col - w / 2) * scale
            y = (h / 2 - row) * scale
            t.goto(x, y)

    turtle.update()
    turtle.done()

if __name__ == '__main__':
    image_shape = Image.shape

    # 1. Convolve with line detection kernels
    kernels = [kernel_h, kernel_v, kernel_45p, kernel_45m]
    line_mask = line_detection_convolution(Image, kernels, threshold=1)

    # 2. Track lines in the binary mask
    paths = track_lines_in_mask(line_mask)

    # 3. Call the drawing function.
    draw_paths(paths, image_shape, scale=1)
