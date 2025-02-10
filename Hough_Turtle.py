from Pre_Turtle import *
import turtle
import networkx as nx

Image = skeleton_cleaned

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
