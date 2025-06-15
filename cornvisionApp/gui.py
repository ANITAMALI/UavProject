from components import CornProgressBar
from gui_events import *
from types import SimpleNamespace

class myApp:
    def __init__(self, root):
        self.root = root
        self.images = {}
        self.images["icon"] = load_PhotoImage("C:\\Users\\Boaz\\Desktop\\University Courses\\D\\Project\\TRY1\\cornvisionApp\\assets\\logo_ico.ico")
        self.images["logo"], self.logo_h = load_PhotoImage("C:\\Users\Boaz\Desktop\\University Courses\D\Project\TRY1\cornvisionApp\\assets\\logo.png", 180)
        self.images["welcome"], self.welcome_h = load_PhotoImage("C:\\Users\Boaz\Desktop\\University Courses\D\Project\TRY1\cornvisionApp\\assets\\Picture5.png", 650)
        self.images["instructions"] = Image.open("C:\\Users\Boaz\Desktop\\University Courses\D\Project\TRY1\cornvisionApp\\assets\\instructions.png")
        self.original_img = Image.open(
            "C:\\Users\Boaz\Desktop\\University Courses\D\Project\TRY1\cornvisionApp\\assets\\background.png")
        self.instructions_tk = self.images["instructions"]

        self.root.iconphoto = (True, self.images["icon"])  # Set the window icon
        self.instructions_popup = None
        self.title = "CornVision App"
        self.canvas = tk.Canvas(root)
        self.canvas.pack(fill="both", expand=True)
        self.bg_img_tk = None
        self.bg_id = None
        self.root.bind("<Configure>", self._resize_bg)
        self.loaded_images = []  # List to store uploaded images

        #self.images["instruction"], _ = self._load_image(".../instructions.png", 500)
        ## --- Logo Section --- ##

        self.label1 = tb.Label(self.canvas, image=self.images["logo"])
        self.logo_window = self.canvas.create_window(self.root.winfo_width() // 2,
                                                     50, window=self.label1, anchor="n", tags="bg") # Center logo on canvas, create a widget of type Label (Create_window creates a window on the canvas)

        self.label2 = tb.Label(self.canvas, image=self.images["welcome"])
        self.welcome_window = self.canvas.create_window(self.root.winfo_width() // 2,
                                                        50 + self.images["logo"].height() + 20, window=self.label2, anchor="n") # Center welcome image on canvas, create a widget of type Label

        self.upload_btn = CornButton(
            parent=self.canvas,
            text="העלאת תמונות",
            style='my.TButton',  # Use a custom style defined in my_styles.py
            command=lambda: upload_and_calculate(self, self.loaded_images),
            cursor="hand2",  # Change cursor to hand when hovering
            width=20,
            height=30)
        self.upload_btn_window = self.canvas.create_window(0, 0, window=self.upload_btn.widget(), anchor="n") # Create a window on the canvas for the button


        self.how_btn = CornButton(
            parent=self.canvas,
            text="?איך זה עובד",
            style='my.TButton',  # Use a custom style defined in my_styles.py
            command=lambda: show_instructions(self),  # define this method in your class
            cursor="hand2",  # Change cursor to hand when hovering
            width=20,
            height=30)
        self.how_btn_window = self.canvas.create_window(0, 0, window=self.how_btn.widget(), anchor="n") # Create a window on the canvas for the how it works button

        self.progress_frame = tk.Frame(self.canvas)
        # Create progress bar inside the frame
        self.progress_bar = CornProgressBar(
            parent=self.progress_frame,
            length=500,
            style='Custom.Horizontal.TProgressbar'
        )
        self.progress_bar.widget().pack(pady=5)  # Add some padding

        # Create text label below progress bar
        self.progress_text = tk.Label(
            self.progress_frame,
            text="...טוען תמונות",
            font=("Calibri", 12)
        )
        self.progress_text.pack(pady=5)  # Add some padding

        self.progress_window = self.canvas.create_window(0, 0, window=self.progress_frame,anchor="n", tags="bar")  # Create a window on the canvas for the progress bar
        self.canvas.itemconfigure(self.progress_window, state='hidden')  # Hide progress bar initially
        self._resize_bg(SimpleNamespace(widget=self.root,
                                        width=self.root.winfo_width(),
                                        height=self.root.winfo_height()))

    def _resize_bg(self, event):
        if event.widget != self.root:
            return
        # Resize background image
        resized = self.original_img.resize((event.width, event.height), Image.LANCZOS)
        self.bg_img_tk = ImageTk.PhotoImage(resized)
        # Update or create image on canvas
        if self.bg_id is None:
            self.bg_id = self.canvas.create_image(0, 0, image=self.bg_img_tk, anchor="nw", tags="logo")
        else:
            self.canvas.itemconfig(self.bg_id, image=self.bg_img_tk, tags="logo")
        # Reposition widgets
        logo_y = 30 # Logo y offset
        welcome_y = self.logo_h + logo_y + 20 # Welcome image y offset
        button1_y = welcome_y + self.welcome_h + 30 # Upload button y offset
        button2_y = button1_y + 140 # Second button y offset (if any)
        canvas_width = event.width # Get current canvas width
        progressbar_y = button2_y + 150 # Progress bar y offset
        self.canvas.coords(self.logo_window, canvas_width // 2, logo_y) # Center logo on canvas
        self.canvas.coords(self.welcome_window, canvas_width // 2, welcome_y) # Center welcome image on canvas
        self.canvas.coords(self.upload_btn_window, canvas_width // 2, button1_y) # Center upload button on canvas
        self.canvas.coords(self.how_btn_window, canvas_width // 2, button2_y)
        self.canvas.coords(self.progress_window, canvas_width // 2, progressbar_y)
        # Force initial resize
        self.canvas.update_idletasks()


