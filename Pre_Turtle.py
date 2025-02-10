#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, ndimage
from skimage import morphology
import networkx as nx
import turtle
import cv2
#%%
# Load the retinal fundus image in grayscale
image_path = 'Fundus.jpeg'
img = cv2.imread(image_path)
if img is None:
    raise IOError("Image not found. Check the image path.")

# Convert to RGB (OpenCV loads images in BGR format by default)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

#%%
def apply_median_filter(image, kernel_size=5):
    """
    Applies a median filter to reduce noise in fundus images.

    Parameters:
        image (numpy.ndarray): Input fundus image (BGR or grayscale)
        kernel_size (int): Size of the median filter kernel (must be odd and >1)

    Returns:
        numpy.ndarray: Filtered image
    """
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be an odd integer")

    # Apply median filter
    filtered_image = cv2.medianBlur(image, kernel_size)

    return filtered_image
#%%
# Perform built-in bilateral (OpenCV)
gray_img_bilateral = cv2.bilateralFilter(gray_img, d=5, sigmaColor=50, sigmaSpace=0)

# Apply median filter with 5x5 kernel
gray_img_median = apply_median_filter(gray_img, kernel_size=5)

#%%
# Create a CLAHE object for comparison
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
clahe_median = clahe.apply(gray_img_median)
clahe_bilateral = clahe.apply(gray_img_bilateral)

#%%
def perona_malik_diffusion(img, num_iterations, kappa, gamma):
    """
    Perform Perona–Malik anisotropic diffusion on a 2D grayscale image.

    Parameters
    ----------
    img            : 2D numpy array (grayscale image)
    num_iterations : int, number of diffusion iterations
    kappa          : float, edge threshold parameter
    gamma          : float, time-step size (should be <= 0.25 for stability in 2D)

    Returns
    -------
    diffused : 2D numpy array, the filtered image
    """

    # Convert to float32 for numerical stability.
    diffused = img.astype(np.float32)

    for _ in range(num_iterations):
        # Compute finite differences (gradients) in the four directions:
        # North gradient (top neighbor)
        gradN = np.roll(diffused, 1, axis=0) - diffused
        # South gradient (bottom neighbor)
        gradS = np.roll(diffused, -1, axis=0) - diffused
        # East gradient (right neighbor)
        gradE = np.roll(diffused, -1, axis=1) - diffused
        # West gradient (left neighbor)
        gradW = np.roll(diffused, 1, axis=1) - diffused

        # Perona–Malik conduction coefficients in each direction.
        # conduction function: c = exp( - (|gradI| / kappa)^2 )
        cN = np.exp(-(gradN / kappa) ** 2)
        cS = np.exp(-(gradS / kappa) ** 2)
        cE = np.exp(-(gradE / kappa) ** 2)
        cW = np.exp(-(gradW / kappa) ** 2)

        # Update the image by discrete PDE:
        diffused += gamma * (
                cN * gradN + cS * gradS +
                cE * gradE + cW * gradW
        )

    return diffused
#%%
niter = 10  # Too few iterations might not remove enough noise; too many can over-smooth or produce artifacts (especially if gamma is large)
kappa = 60  # If you see too much blurring at vessel edges, lower kappa. If you see little noise reduction, increase kappa.
gamma = 0.1  # If you see “ringing” or instability, lower gamma. If you want the same smoothing in fewer iterations, you can raise gamma

# Apply our pure NumPy Perona–Malik filter
Aniso_img = perona_malik_diffusion(clahe_median,
                                     num_iterations=niter,
                                     kappa=kappa,
                                     gamma=gamma)

#%%
def create_LoG_kernel(kernel_size, sigma):
    """
    Create a Laplacian of Gaussian (LoG) kernel.

    Equation:
        LoG(x, y) = (1 / (pi * sigma^4)) * (1 - ((x^2 + y^2) / (2 * σ^2))) * exp(-(x² + y²) / (2σ²))

    Parameters:
        kernel_size (int): Size of the kernel (must be an odd number).
        sigma (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: LoG kernel.
    """
    # Create coordinate grid centered at 0
    ax = np.arange(-(kernel_size // 2), kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    norm2 = xx**2 + yy**2

    # Compute the LoG function
    factor = 1 / (np.pi * sigma**4)
    kernel = factor * (1 - norm2 / (2 * sigma**2)) * np.exp(-norm2 / (2 * sigma**2))

    # Normalize kernel to ensure zero-sum
    kernel_mean = kernel.mean()
    kernel = kernel - kernel_mean

    return kernel
#%%
sigmas_LoG = [0.5, 1, 1.5, 2, 2.5, 3]  # 1,1.4,2,2.8
results_LoG = []  # to store convolved images for each sigma

for sigma in sigmas_LoG:
    kernel_size = int(np.ceil(6 * sigma))  # Rule of thumb for kernel size
    if kernel_size % 2 == 0:
        kernel_size += 1  # Ensure odd size
    kernel_LoG = create_LoG_kernel(kernel_size, sigma)

    # Convolve
    convolved_LoG = signal.convolve2d(Aniso_img, kernel_LoG, boundary='symm', mode='same')

    # Normalize to [0, 255] for display
    # NOTE: LoG can have negative values; here we do a simple linear shift.
    min_val = convolved_LoG.min()
    max_val = convolved_LoG.max()
    convolved_LoG -= min_val
    if (max_val - min_val) > 1e-12:
        convolved_LoG *= (255.0 / (max_val - min_val))

    results_LoG.append((sigma, convolved_LoG))

# 3) Plot only the convolved images for each sigma
num_scales = len(sigmas_LoG)
#%%
def zero_crossing_morphological(log_image):
    """
    Return a binary edge map where zero-crossings occur in `log_image`.
    This is a simple morphological approach:
     - Convert LoG to a binary mask: foreground = log_image > 0
     - Mark boundary pixels: any foreground pixel adjacent to a background pixel (or vice versa).
    """
    # Foreground (positive) vs. background (non-positive)
    foreground = (log_image > 0)

    # Erode (Shrink) foreground
    structure = np.ones((3,3), dtype=bool)
    eroded_fg = ndimage.binary_erosion(foreground, structure=structure)

    # The boundary = where the foreground mask differs from its eroded version
    # (this catches "outer edges" of the foreground). But we also want the boundary
    # from background side, so we do the same with background or do an XOR with neighbors.
    edge_map_fg = foreground ^ eroded_fg  # boundary from the "foreground" side
    background = ~foreground
    eroded_bg = ndimage.binary_erosion(background, structure=structure)
    edge_map_bg = background ^ eroded_bg  # boundary from the "background" side

    edge_map = edge_map_fg | edge_map_bg  # Logical OR operator
    return edge_map

def zero_crossing_min_abs(log_image, threshold=0.03):
    """
    Standard zero-crossing detection:
    1. Check for sign changes in 8-connected neighbors.
    2. Apply a threshold on the minimum magnitude difference.
    """
    H, W = log_image.shape
    edge_map = np.zeros((H, W), dtype=bool)

    # Offsets for 8-connected neighbors
    neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                   (0, -1),          (0, 1),
                   (1, -1), (1, 0), (1, 1)]

    for y in range(H):
        for x in range(W):
            current_val = log_image[y, x]
            current_sign = current_val > 0

            for dy, dx in neighbors_8:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W:
                    neighbor_val = log_image[ny, nx]
                    neighbor_sign = neighbor_val > 0

                    # Check for sign change
                    if current_sign != neighbor_sign:
                        # Threshold on magnitude difference
                        if abs(current_val) + abs(neighbor_val) > threshold:
                            edge_map[y, x] = True
                            break  # Mark once per pixel

    return edge_map
#%%
# 1) Equalize each scale so each contributes similarly
#    Then recenter so that 0 stays 0-ish.
#    We'll do: val -> (val - min) / (max-min) in [0,1], then shift by -0.5.
eq_images = []
for (sigma, log_img) in results_LoG:
    min_val = log_img.min()
    max_val = log_img.max()
    # To avoid division by zero
    if (max_val - min_val) < 1e-12:
        eq = np.zeros_like(log_img)
    else:
        eq = (log_img - min_val) / (max_val - min_val)
    # shift so we have negative/positive range
    eq = eq - 0.5
    eq_images.append(eq)

# 2) Summation or averaging
#    We'll take the mean across scales (axis=0 merges them along H/W).
sum_image = np.mean(eq_images, axis=0)

# 3) Zero-crossing detection
#    a) Simple morphological
edges_morph = zero_crossing_morphological(sum_image)
#    b) "Lowest absolute magnitude" approach
edges_minabs = zero_crossing_min_abs(sum_image, threshold=0.028)

#%%
# Assume 'zero_crossing' is your binary image after zero-crossing detection
selem = morphology.disk(4)  # Small structuring element

# Expand the image (dilation)
expanded = morphology.binary_dilation(edges_minabs, selem)

# Shrink the image (erosion)
shrunk = morphology.binary_erosion(expanded, selem)

cleaned_edges = morphology.remove_small_objects(edges_morph, min_size=100)

#%%
skeleton_cleaned = morphology.skeletonize(cleaned_edges)
skeleton_morph = morphology.skeletonize(shrunk)

#%%
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
    """
    img: grayscale image (numpy array).
    kernels: list of convolution kernels.
    threshold: minimum response to keep as 'line'.
    Returns a binary mask of 'line' pixels.
    """
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
    line_mask = (max_resp > threshold).astype(np.uint8)
    return line_mask

def track_lines_in_mask(line_mask):
    """
    Naive approach to track continuous lines in the thresholded mask.
    We'll do a flood-fill / DFS. Each fill becomes one 'path'.
    """
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
                # push neighbors that are line pixels
                for nr, nc in neighbors(rr, cc):
                    if line_mask[nr, nc] == 1 and not visited[nr, nc]:
                        stack.append((nr, nc))
        return path_pixels

    for row in range(h):
        for col in range(w):
            if line_mask[row, col] == 1 and not visited[row, col]:
                path = dfs(row, col)
                if len(path) > 500:  # ignore single-pixel
                    paths.append(path)

    return paths

# 1. Convolve with line detection kernels
kernels = [kernel_h, kernel_v, kernel_45p, kernel_45m]
line_mask = line_detection_convolution(skeleton_cleaned, kernels, threshold=1)

# 2. Track lines in the binary mask
paths = track_lines_in_mask(line_mask)

#%%
