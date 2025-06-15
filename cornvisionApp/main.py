import ttkbootstrap as tb
from gui import myApp
from my_styles import apply_custom_styles


if __name__ == "__main__":
    app = tb.Window(themename="flatly", title="CornVision")
    app.geometry("1080x920")
    app.minsize(1000, 800)
    apply_custom_styles()
    myApp(app)
    app.mainloop()