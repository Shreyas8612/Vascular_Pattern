from Pre_Turtle import *
from Pre_Turtle_2 import *
import turtle

# Import the skeleton image from the previous step
Image = skeleton_cleaned  # For gradient operator - change it to skeleton and min path length to 1
scale_kernel = 4

# Horizontal
kernel_h = np.array([[-1,  2, -1],
                     [-1,  2, -1],
                     [-1,  2, -1]], dtype=np.float32) * scale_kernel

# Vertical
kernel_v = np.array([[-1, -1, -1],
                     [ 2,  2,  2],
                     [-1, -1, -1]], dtype=np.float32) * scale_kernel

# +45 degree
kernel_45p = np.array([[ 2, -1, -1],
                       [-1,  2, -1],
                       [-1, -1,  2]], dtype=np.float32) * scale_kernel

# -45 degree
kernel_45n = np.array([[-1, -1,  2],
                       [-1,  2, -1],
                       [ 2, -1, -1]], dtype=np.float32) * scale_kernel

kernel_names = {
    "kernel_h": kernel_h,
    "kernel_v": kernel_v,
    "kernel_45p": kernel_45p,
    "kernel_45n": kernel_45n
}

def line_detection_convolution(img, kernels, threshold=1):
    h, w = img.shape
    num_kernels = len(kernels)

    # Initialize response stack
    response_stack = np.zeros((h, w, num_kernels), dtype=np.float32)

    # Apply each kernel to the image
    for i in range(num_kernels):
        kernel = kernels[i]
        response = signal.convolve2d(img, kernel, boundary='symm', mode='same')
        response_stack[:, :, i] = response  # Store response for the current kernel

    # Compute maximum response across all kernels
    max_resp = np.max(response_stack, axis=2)

    # Apply thresholding
    line_mask = (max_resp > threshold).astype(np.float32)

    return response_stack, line_mask, max_resp


def plot_response_stack(response_stack, kernels):
    num_kernels = len(kernels)

    # Get corresponding kernel names
    kernel_labels = [name for name, k in kernel_names.items()]

    # Create figure with subplots
    fig, axes = plt.subplots(1, num_kernels, figsize=(15, 5))

    # Iterate through each response
    for i in range(num_kernels):
        ax = axes[i] if num_kernels > 1 else axes  # Handle case when there's only one kernel
        response = response_stack[:, :, i]  # Extract response for current kernel
        ax.imshow(response, cmap='gray')
        ax.set_title(f'{kernel_labels[i]} Response')
        ax.axis('off')

    plt.tight_layout()
    plt.show()
    return fig

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
    kernels = [kernel_h, kernel_v, kernel_45p, kernel_45n]
    response, line_mask, max_resp = line_detection_convolution(Image, kernels, threshold=0)

    # 2. Plot the response stack
    plot = plot_response_stack(response, kernels)

    # 3. Track lines in the binary mask
    paths = track_lines_in_mask(line_mask)

    # 4. Display the results
    plt.figure(figsize=(20, 7))
    # Display the line mask, max response, and skeleton image with tracked paths
    for i, (img, title) in enumerate([ (max_resp, "Max Response"), (line_mask, "Line Mask")]):
        plt.subplot(1, 3, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')

    # Third subplot: Skeleton image with tracked paths
    plt.subplot(1, 3, 3)
    plt.imshow(Image, cmap='gray')
    for path in paths:
        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], marker='.', linestyle='-', linewidth=1.2, markersize=1, alpha=0.8)

    plt.title("Tracked Paths Over Skeleton Image")
    plt.axis('off')

    plt.tight_layout()  # Ensures spacing between subplots
    plt.show()

    # 5. Draw the paths using turtle graphics
    draw_paths(paths, image_shape, scale=1)
