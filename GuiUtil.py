"""
GuiUtil gets a basic GUI going.
It is based on https://stackoverflow.com/questions/17466561/best-way-to-structure-a-tkinter-application

"""
import tkinter as tk
import time

class Label(tk.Frame):
    def __init__(self, parent, label, *args, **kwargs):
        msg = tk.Label(root, text=label)
        msg.config(bg='lightgreen', font=('times', 24, 'italic'))
        msg.pack()


class GuiUtil(tk.Frame):
    def __init__(self, parent, label:str=None, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        if label:
            self.label = Label(parent=self, label=label)

    @staticmethod
    def set_clipboard(text_to_set: str):
        r = tk.Tk()
        r.withdraw()
        r.clipboard_clear()
        r.clipboard_append(text_to_set)

        r.update()
        # time.sleep(.2)
        # r.update()

        r.destroy()

    @staticmethod
    def get_clipboard() -> str:
        root = tk.Tk()
        root.withdraw()
        return root.clipboard_get()

if __name__ == "__main__":
    root = tk.Tk()
    GuiUtil(root, label="Hi!").pack(side="top", fill="both", expand=True)
    root.mainloop()