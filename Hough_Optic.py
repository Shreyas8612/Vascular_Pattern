from Pre_Turtle import *
# from Pre_Turtle_2 import *
import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from skimage import io, color, feature, transform
import turtle

# -----------------------------
# Step 1: Optic disc detection using Hough transform.
def detect_optic_nerve_center(image):

    # Define a range for expected optic disc radii
    hough_radii = np.arange(55, 100, 2)

    # Compute the Hough transform accumulator for circles
    hough_res = transform.hough_circle(image, hough_radii)

    # Extract the most prominent circle (i.e., the peak in the accumulator)
    accums, cx, cy, radii = transform.hough_circle_peaks(hough_res, hough_radii,
                                                         total_num_peaks=1)
    if len(cx) > 0:
        center = (cx[0], cy[0])
        radius = radii[0]

        # Visualize the result
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(image, cmap='gray')
        circ = plt.Circle(center, radius, color='red', fill=False, linewidth=2)
        ax.add_patch(circ)
        ax.plot(center[0], center[1], 'bo', markersize=5)  # mark center
        ax.set_title("Optic Disc Detection")
        ax.axis('off')
        plt.show()
        return center
    else:
        print("No optic disc detected.")
        return None

# -----------------------------
# Step 2: Line detection via convolution.
scale_kernel = 4

kernel_h = np.array([[-1, 2, -1],
                     [-1, 2, -1],
                     [-1, 2, -1]], dtype=np.float32) * scale_kernel

kernel_v = np.array([[-1, -1, -1],
                     [2, 2, 2],
                     [-1, -1, -1]], dtype=np.float32) * scale_kernel

kernel_45p = np.array([[2, -1, -1],
                       [-1, 2, -1],
                       [-1, -1, 2]], dtype=np.float32) * scale_kernel

kernel_45n = np.array([[-1, -1, 2],
                       [-1, 2, -1],
                       [2, -1, -1]], dtype=np.float32) * scale_kernel

kernels = [kernel_h, kernel_v, kernel_45p, kernel_45n]


def line_detection_convolution(img, kernels, threshold=1):
    h, w = img.shape
    num_kernels = len(kernels)
    response_stack = np.zeros((h, w, num_kernels), dtype=np.float32)
    for i in range(num_kernels):
        kernel = kernels[i]
        response = signal.convolve2d(img, kernel, boundary='symm', mode='same')
        response_stack[:, :, i] = response
    max_resp = np.max(response_stack, axis=2)
    line_mask = (max_resp > threshold).astype(np.float32)
    return response_stack, line_mask, max_resp


def plot_response_stack(response_stack, kernels):
    kernel_labels = ["kernel_h", "kernel_v", "kernel_45p", "kernel_45n"]
    num_kernels = len(kernels)
    fig, axes = plt.subplots(1, num_kernels, figsize=(15, 5))
    for i in range(num_kernels):
        ax = axes[i] if num_kernels > 1 else axes
        response = response_stack[:, :, i]
        ax.imshow(response, cmap='gray')
        ax.set_title(f'{kernel_labels[i]} Response')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    return fig


# -----------------------------
# Step 3: Track lines in the binary mask via DFS.
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

    def dfs(r, c):
        stack = [(r, c)]
        path_pixels = []
        while stack:
            rr, cc = stack.pop()
            if not visited[rr, cc]:
                visited[rr, cc] = True
                path_pixels.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if line_mask[nr, nc] == 1 and not visited[nr, nc]:
                        stack.append((nr, nc))
        return path_pixels

    for row in range(h):
        for col in range(w):
            if line_mask[row, col] == 1 and not visited[row, col]:
                path = dfs(row, col)
                if len(path) > 500:  # keep only long paths
                    paths.append(path)
    return paths


# -----------------------------
# Step 4: Draw the detected paths using turtle.
def draw_paths(paths, image_shape, scale=1):
    h, w = image_shape
    screen = turtle.Screen()
    screen.setup(width=w * scale, height=h * scale)
    # Center coordinate system: (0,0) at screen center.
    screen.setworldcoordinates(-w / 2 * scale, -h / 2 * scale, w / 2 * scale, h / 2 * scale)
    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    t.penup()
    turtle.tracer(0, 0)
    for path in paths:
        if not path:
            continue
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


# -----------------------------
# Main script
if __name__ == "__main__":
    # Assume 'Aniso_img' (grayscale fundus image) and 'skeleton_cleaned' (binary vessel skeleton) are imported from Pre_Turtle.
    image_shape = skeleton_cleaned.shape

    # Detect the optic nerve center.
    center = detect_optic_nerve_center(cleaned_edges)
    print("Optic nerve center (row, col):", center)

    # Use convolution-based line detection on the skeleton image.
    response, line_mask, max_resp = line_detection_convolution(skeleton_cleaned, kernels, threshold=0)

    # Track continuous lines in the binary mask.
    paths = track_lines_in_mask(line_mask)
    print(f"Tracked {len(paths)} long paths from the line mask.")

    # Draw the paths using turtle graphics.
    draw_paths(paths, image_shape, scale=1)
