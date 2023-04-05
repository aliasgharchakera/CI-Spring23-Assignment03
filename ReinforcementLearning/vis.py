from tkinter import *
from PIL import ImageTk as tk, Image



class DisplayGrid:

    def __init__(self, n:int, redBlock:list, green, arrowDic:dict):
        self.size = n
        self.redBlocks = redBlock
        self.green = green
        self.arrowDic = arrowDic

    def createWindow(self):
        self.window = Tk()
        self.window.title("Image Grid")
        self.canvas = Canvas(self.window, width=700, height=700, bg="white")
        self.canvas.pack()
        cell_size = 700 // self.size
        self.imagePath = {'up':tk.PhotoImage(file="images/up_arrow.png"), 'down':tk.PhotoImage(file="images/down_arrow.png"), 'left':tk.PhotoImage(file="images/left_arrow.png"), 'right': tk.PhotoImage(file="images/right_arrow.png")}


# Place the images on the canvas at the specified coordinates
        for i in range(self.size):
            for j in range(self.size):
                x0, y0 = i * cell_size, j * cell_size
                x1, y1 = x0 + cell_size, y0 + cell_size
                if (j, i) in self.redBlocks:
                    # canvas.create_image(x0, y0, anchor=NW, image=images[coordinates.index((i, j))])
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="red")
                elif (j,i) in self.green:
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill="green")
                else:
                    if self.arrowDic.get((j,i)):
                        # self.img = self.img = ImageTk.PhotoImage(file= self.imagePath[self.arrowDic[(j,i)]])
                        self.canvas.create_image(x0, y0, anchor=NW, image= self.imagePath[self.arrowDic[(j,i)]])
                        # self.canvas.create_rectangle(x0, y0, x1, y1, fill="white")
                    else:
                        self.canvas.create_rectangle(x0, y0, x1, y1, fill="white")
        self.window.mainloop()

