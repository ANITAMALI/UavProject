import ttkbootstrap as tb
import tkinter as tk
import os
from tkinter import filedialog
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD
import run_inferrence_on_images
import run_prediction
import numpy as np
import cv2 as cv
# --- Global State ---
loaded_images = []
uploaded_paths = []
selected_image_index = None
show_images = True

# --- GUI Functions ---

def upload_and_show():
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_paths:
        return

    for path in file_paths:
        if path in uploaded_paths:
            continue  # Skip duplicates

        img = Image.open(path)
        img_width = 150
        aspect_ratio = img.height / img.width
        img_height = int(img_width * aspect_ratio)
        resized_img = img.resize((img_width, img_height))
        tk_img = ImageTk.PhotoImage(resized_img)
        loaded_images.append({
            "tk_img": tk_img,
            "height": img_height,
            "original_pil": img,
            "file_path": path
        })
        uploaded_paths.append(path)
        print(loaded_images)
    layout_images()


def layout_images():
    for widget in image_container.winfo_children():
        widget.destroy()

    if not loaded_images:
        return

    max_width = image_canvas.winfo_width()
    padding = 10
    x, y = padding, padding
    row_height = 0

    for idx, img_data in enumerate(loaded_images):
        img_width = 150
        img_height = img_data["height"]
        tk_img = img_data["tk_img"]
        file_name = os.path.basename(img_data["file_path"])

        img_frame = tk.Frame(image_container, bd=1, relief="flat")

        # Image name above
        name_label = tk.Label(img_frame, text=file_name, wraplength=140, font=("Helvetica", 8), anchor="center")
        name_label.pack(pady=(0, 2))

        # Image
        img_label = tk.Label(img_frame, image=tk_img)
        img_label.pack(fill="both", expand=True, pady=10)
        img_label.bind("<Double-Button-1>", lambda e, i=idx: zoom_image(i))

        img_frame.place(x=x, y=y)
        x += img_width + padding
        row_height = max(row_height, img_height + 30)

        if x + img_width > max_width:
            x = padding
            y += row_height + padding
            row_height = 0

    total_height = y + row_height + padding
    image_container.config(width=max_width, height=total_height)
    image_canvas.config(scrollregion=image_canvas.bbox("all"))


def zoom_image(start_index):
    if not loaded_images:
        return

    zoom_win = tk.Toplevel(app)
    zoom_win.title("Zoom Viewer")
    zoom_win.geometry("850x700")  # Initial size, adjusted dynamically below

    # Scrollable canvas
    zoom_canvas = tk.Canvas(zoom_win)
    zoom_canvas.pack(side='left', fill='both', expand=True)

    # 👇 Bind mouse wheel scroll
    zoom_canvas.bind("<Enter>", lambda e: zoom_canvas.bind_all("<MouseWheel>", lambda e: zoom_canvas.yview_scroll(
        -1 * int(e.delta / 120), "units")))
    zoom_canvas.bind("<Leave>", lambda e: zoom_canvas.unbind_all("<MouseWheel>"))

    v_scrollbar = tb.Scrollbar(zoom_win, orient='vertical', command=zoom_canvas.yview)
    v_scrollbar.pack(side='right', fill='y')

    zoom_canvas.configure(yscrollcommand=v_scrollbar.set)
    zoom_canvas.bind('<Configure>', lambda e: zoom_canvas.itemconfig("frame", width=e.width))

    image_frame = tk.Frame(zoom_canvas)
    canvas_window = zoom_canvas.create_window((0, 0), window=image_frame, anchor='nw', tags="frame")

    def on_frame_configure(event):
        zoom_canvas.configure(scrollregion=zoom_canvas.bbox("all"))

    image_frame.bind("<Configure>", on_frame_configure)

    img_label = tk.Label(image_frame)
    img_label.pack(pady=10)

    btn_row = tb.Frame(image_frame)
    btn_row.pack(pady=10)

    def update_view(index):
        nonlocal current_index
        current_index = index % len(loaded_images)
        img_data = loaded_images[current_index]
        img = img_data["original_pil"]

        zoom_width = 800
        aspect_ratio = img.height / img.width
        zoom_height = int(zoom_width * aspect_ratio)
        resized = img.resize((zoom_width, zoom_height), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(resized)

        img_label.config(image=tk_img)
        img_label.image = tk_img
        zoom_win.title(os.path.basename(img_data["file_path"]))

    def next_image():
        update_view(current_index + 1)

    def prev_image():
        update_view(current_index - 1)

    def close_window():
        zoom_win.destroy()

    def delete_window():
        nonlocal current_index
        if current_index >= len(loaded_images):
            zoom_win.destroy()
            return
        del loaded_images[current_index]
        del uploaded_paths[current_index]
        layout_images()

        if loaded_images:
            if current_index >= len(loaded_images):
                current_index = len(loaded_images) - 1
            update_view(current_index)
        else:
            zoom_win.destroy()

    current_index = start_index
    update_view(current_index)

    # Buttons
    prev_btn = tb.Button(btn_row, text="← Prev", bootstyle="secondary", command=prev_image)
    prev_btn.pack(side="left", padx=10)

    next_btn = tb.Button(btn_row, text="Next →", bootstyle="secondary", command=next_image)
    next_btn.pack(side="left", padx=10)

    delete_btn = tb.Button(btn_row, text="Delete", bootstyle="danger", command=delete_window)
    delete_btn.pack(side="left", padx=10)

    close_btn = tb.Button(btn_row, text="Close", bootstyle="secondary", command=close_window)
    close_btn.pack(side="left", padx=10)




def toggle_images():
    global show_images
    show_images = not show_images
    if show_images:
        image_canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        toggle_btn.configure(text="Hide Uploaded Images")
    else:
        image_canvas.pack_forget()
        scrollbar.pack_forget()
        toggle_btn.configure(text="Show Uploaded Images")


def handle_drop(event):
    paths = app.tk.splitlist(event.data)
    valid_extensions = (".jpg", ".jpeg", ".png")
    for path in paths:
        if not path.lower().endswith(valid_extensions):
            continue
        if path in uploaded_paths:
            continue
        try:
            img = Image.open(path)
            img_width = 150
            aspect_ratio = img.height / img.width
            img_height = int(img_width * aspect_ratio)
            resized_img = img.resize((img_width, img_height))
            tk_img = ImageTk.PhotoImage(resized_img)
            loaded_images.append({
                "tk_img": tk_img,
                "height": img_height,
                "original_pil": img,
                "file_path": path
            })
            uploaded_paths.append(path)
        except Exception as e:
            print(f"Error loading {path}:", e)
    layout_images()


def run_model_on_all_images(): # Need to finalize this function
    if not loaded_images:
        return
    elif len(loaded_images) <= 1:
        tk.messagebox.showwarning("wa", "Please upload at least 2 images to run the model.")
    else:
        print("Running model on all loaded images...")
        pil_images = [img["original_pil"] for img in loaded_images] # Extract original PIL images
        opencv_images = [cv.cvtColor(np.array(pil), cv.COLOR_RGB2BGR) for pil in pil_images] # Convert PIL images to OpenCV format
        results = run_prediction.run_yolo_on_images(opencv_images)
    return results

### --- GUI Setup --- ###
app = TkinterDnD.Tk()  # create base window
style = tb.Style("superhero")  # Manually apply ttkbootstrap theme
style.master = app  # Apply theme to the existing app window
app.title("Selectable Image Viewer") # Set the title of the app window
app.geometry("900x600") # Set the initial size of the app window
app.minsize(650,450) # Set the minimum size of the app window

## --- Top Section --- ##
top_frame = tb.Frame(app) # Create top section Frame widget
top_frame.pack(pady=10) # Add some padding around the top section,
# Pack Arranges widgets in blocks before placing them in the parent.

# --- Top Section Buttons Frame and Buttons --- #
button_row = tb.Frame(top_frame)
button_row.pack()
upload_btn = tb.Button(button_row, text="Upload Images", bootstyle="infooutline", command=upload_and_show)
upload_btn.pack(pady=4, padx=4, side="left")
run_btn = tb.Button(button_row, text="Run Recognition", bootstyle="success", command=run_model_on_all_images)
run_btn.pack(pady=4, padx=4, side="left")


# --- Separator --- #
separator = tb.Separator(app, orient='horizontal')
separator.pack(fill='x', pady=10)


'''Layered Structure Below Explained: 
    1. canvas_frame = tb.Frame(app) - This is just a container for everything - Canvas, Scrollbar, Buttons
    2. image_canvas = tk.Canvas(canvas_frame) - The Canvas is scrollable — unlike a Frame. But! Canvas alone can only 
    display basic shapes or embedded windows
    3. image_container = tk.Frame(image_canvas) - We create a Frame inside the Canvas. This is the trick:
    We can now use pack(), place(), or grid() normally inside this inner Frame.
    4. image_canvas.create_window((0, 0), window=image_container, anchor='nw') 
    - The result: a scrollable layout-capable container.'''

# --- Uploaded Images Frame + Toggle button --- #
canvas_frame = tb.Frame(app, height=100)
canvas_frame.pack(fill='both', expand=True)
toggle_btn = tb.Button(canvas_frame, text="Hide Uploaded Images", command=toggle_images)
toggle_btn.pack(pady=4)

image_canvas = tk.Canvas(canvas_frame, highlightthickness=0)
image_canvas.pack(side='left', fill='both', expand=True)

scrollbar = tb.Scrollbar(canvas_frame, orient='vertical', command=image_canvas.yview) #Create scrollbar
scrollbar.pack(side='right', fill='y')
image_canvas.configure(yscrollcommand=scrollbar.set)

image_canvas.bind("<Enter>", lambda e: image_canvas.bind_all("<MouseWheel>", lambda e: image_canvas.yview_scroll(-1 * int(e.delta / 120), "units")))
image_canvas.bind("<Leave>", lambda e: image_canvas.unbind_all("<MouseWheel>"))


image_container = tk.Frame(image_canvas)
image_canvas.create_window((0, 0), window=image_container, anchor='nw')

image_canvas.bind("<Configure>", lambda e: layout_images()) #Bind the scroller to the Frame

# --- Separator --- #
separator = tb.Separator(app, orient='horizontal')
separator.pack(fill='x', pady=10)

# Register the app window for file drops
app.drop_target_register(DND_FILES)
app.dnd_bind('<<Drop>>', handle_drop)

app.mainloop()
