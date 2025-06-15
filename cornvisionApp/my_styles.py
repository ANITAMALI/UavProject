import ttkbootstrap as tb


def apply_custom_styles():
    style = tb.Style()
    style.configure(
        "my.TButton",
        font=("Calibri", 28, "bold"),
        foreground="white",  # text color
        background="#40633E",
        borderwidth=0,
        padding=10
    )
    style.map(
        "my.TButton",
        background=[('active', '#97CA64')],
        foreground=[('active', 'white')]
    )
    style.map(
        "save.TButton",
        background=[('active', '#97CA64')],
        foreground=[('active', 'white'), ('disabled', 'white')]
    )

    style.configure(
        "save.TButton",
        background='#40633E',
        font=("Calibri", 24, "bold"),
        borderwidth=0,
        padding=10
    )
    style.configure(
        "Custom.Label",
        font=("David", 16, "bold"))

    style.configure(
        "Custom.Horizontal.TProgressbar",
        thickness=20,
        background="#97CA64")  # 25 px tall