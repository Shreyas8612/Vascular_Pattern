from Pre_Turtle import *
import turtle
import numpy as np

def trace_paths_with_turtle(paths, scale=1.0, offset_x=0.0, offset_y=0.0):
    """
    Draw line paths in a turtle window.
    :param paths: list of paths, each path is either:
                    (A) A list of (row, col) pairs, or
                    (B) A NumPy array of shape (N, 2).
    :param scale: how large (or small) to draw each pixel step.
    :param offset_x: shift everything horizontally in turtle coordinates.
    :param offset_y: shift everything vertically in turtle coordinates.
    """

    # 1) Create the screen
    screen = turtle.Screen()
    screen.setup(width=800, height=800)   # pick any reasonable size

    # 2) Create a turtle
    t = turtle.Turtle()
    t.speed('fastest')  # speeds up drawing
    t.color('black')    # change color if you like
    t.pensize(3)        # change pen size if you like
    t.hideturtle()      # hide the turtle icon if you like

    screen.tracer(False)  # only needed if you want to see the drawing process
    screen.update()
    for path in paths:
        # Make sure it's a NumPy array so indexing is easy:
        path_array = np.array(path, dtype=np.float32)

        # If path is too short, skip it
        if len(path_array) < 500:
            continue

        # Move to the first point
        r0, c0 = path_array[0]
        x0 = c0 * scale + offset_x
        y0 = -r0 * scale + offset_y   # negative row => typical "down is increasing r"
        t.penup()
        t.goto(x0, y0)
        t.pendown()

        # Trace the rest
        for (r, c) in path_array[1:]:
            x = c * scale + offset_x
            y = -r * scale + offset_y
            t.goto(x, y)
        # Keep the window open until clicked
        screen.exitonclick()

def main():
    paths = track_lines_in_mask(line_mask)

    # Convert all to arrays if not already:
    for i in range(len(paths)):
        paths[i] = np.array(paths[i], dtype=np.float32)

    # Choose a scale so that the turtle can draw it nicely:
    scale = 1
    # Possibly also shift them so the figure is near the center:
    offset_x = 0.0
    offset_y = 0.0

    trace_paths_with_turtle(paths, scale, offset_x, offset_y)


if __name__ == "__main__":
    main()
