import ttkbootstrap as tb


class CornButton:
    def __init__(self, parent, text, command, style="primary", width=20, height=2, cursor="hand2", **kwargs):
        self.button = tb.Button(
            parent,
            text=text,
            style=style,
            cursor=cursor,
            command=command if command else lambda: None,
            width=width,
            padding=(5, height), # simulate height via padding # Use custom style defined in styles.py
            **kwargs
        )

    def pack(self, **kwargs):
        self.button.pack(**kwargs)

    def grid(self, **kwargs):
        self.button.grid(**kwargs)

    def place(self, **kwargs):
        self.button.place(**kwargs)

    def widget(self):
        return self.button

class CornProgressBar:
    def __init__(self, parent, length=400, mode="determinate", bootstyle="success-striped", style="primary"):
        self.progress = tb.Progressbar(
            parent,
            length=length,
            mode=mode,
            bootstyle=bootstyle,
            value=0,
            maximum=100,
            style=style
        )

    def start(self, interval=20):
        self.progress.start(interval)  # start the animation

    def stop(self):
        self.progress.stop()  # stop animation

    def pack(self, **kwargs):
        self.progress.pack(**kwargs)

    def grid(self, **kwargs):
        self.progress.grid(**kwargs)

    def place(self, **kwargs):
        self.progress.place(**kwargs)

    def widget(self):
        return self.progress



