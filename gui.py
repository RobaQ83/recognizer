from tkinter import (
    Canvas,
    ROUND,
    ALL,
    BOTH,
    Label,
    Frame,
    ttk,
    LEFT,
)

from utils import get_pixels_from

DEFAULT_FOREGROUND_COLOR = "black"
DEFAULT_BACKGROUND_COLOR = "white"
DEFAULT_PEN_WIDTH = 15

DEFAULT_WINDOW_WIDTH = 400
DEFAULT_WINDOW_HEIGHT = 400


class Application:
    def __init__(self, master, callback):
        self.master = master

        self.color_fg = DEFAULT_FOREGROUND_COLOR
        self.color_bg = DEFAULT_BACKGROUND_COLOR

        self.pen_width = DEFAULT_PEN_WIDTH

        self.old_x = None
        self.old_y = None

        self.canvas = Canvas(
            self.master,
            width=DEFAULT_WINDOW_WIDTH,
            height=DEFAULT_WINDOW_HEIGHT,
            bg=self.color_bg,
        )

        self._callback = callback

    def paint(self, e):
        if self.old_x and self.old_y:
            self.canvas.create_line(
                self.old_x,
                self.old_y,
                e.x,
                e.y,
                width=self.pen_width,
                fill=self.color_fg,
                capstyle=ROUND,
                smooth=True,
            )
        self.old_x = e.x
        self.old_y = e.y

    def reset(self, e):
        self.old_x = None
        self.old_y = None

        pixels = get_pixels_from(self.canvas)

        result = self._callback(pixels)
        self.prediction = Label(self.controls, text=f"{result}", font="arial 15").grid(
            row=12, column=0
        )

    def clear(self):
        self.canvas.delete(ALL)

    def _draw_widgets(self):
        self.controls = Frame(self.master, width=500, height=500, padx=5, pady=5)
        Label(self.controls, text="Rozpoznana litera", font="arial 15").grid(row=10, column=0)

        self.prediction = Label(self.controls, text="", font="arial 15").grid(
            row=12, column=0
        )

        self.clearBtn = ttk.Button(self.controls, text="Czyść", command=self.clear)
        self.clearBtn.grid(row=14, column=0, ipadx=30, pady=5)

        self.controls.pack(side=LEFT)

        self.canvas.pack(fill=BOTH, expand=True)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

    def start(self):
        self._draw_widgets()
        self.master.title("Rozpoznawanie liter")
        self.master.resizable(False, False)
        self.master.mainloop()
