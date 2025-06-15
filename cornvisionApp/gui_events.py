import ttkbootstrap as tb
from tkinter import messagebox
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from components import CornButton
from perform_analysis import analyze_images
import cv2 as cv
import numpy as np


def upload_and_calculate(self, loaded_images):
    file_paths = filedialog.askopenfilenames(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if len(file_paths) < 2:
        messagebox.showerror("ארעה שגיאה", "יש להעלות לפחות 2 תמונות")
        return
    if not file_paths:
        return
    self.canvas.itemconfigure(self.progress_window, state='normal')
    for path in file_paths:
        if path in loaded_images:
            continue  # Skip duplicates
        img = Image.open(path)
        img_width = 150
        aspect_ratio = img.height / img.width
        img_height = int(img_width * aspect_ratio)
        resized_img = img.resize((img_width, img_height))
        tk_img = ImageTk.PhotoImage(resized_img)
        loaded_images.append({
            "tk_img": tk_img,
            "original_pil": img,
        })
        #loaded_images.append(path)
        self.progress_bar.progress['value'] += 100 / len(file_paths)  # Update progress bar
        self.root.update_idletasks()  # Process events to update UI
        print(loaded_images)
    self.canvas.after(1000, self.progress_bar.stop())
    self.canvas.after(2000, clear_canvas_show_results(self))

def show_instructions(self):
    if self.instructions_popup and self.instructions_popup.winfo_exists():
        self.instructions_popup.lift()
        return
    self.instructions_popup = tb.Toplevel(self.root, resizable=(False,False))
    self.instructions_popup.title("?איך זה עובד")
    self.instructions_popup.geometry("1280x720")
    self.instructions_popup.configure(background="white")

    img_canvas = tk.Canvas(self.instructions_popup)
    img_canvas.pack(fill="both", expand=True)

    # Initial display
    w, h = 1280, 720
    pil = self.images["instructions"].resize((w, h), Image.LANCZOS)
    tk_img = ImageTk.PhotoImage(pil)
    img_id = img_canvas.create_image(0, 0, image=tk_img, anchor="nw")
    img_canvas.image = tk_img  # Keep reference!

    # Close button
    close_btn = CornButton(
        parent=img_canvas,
        text="סגור",
        cursor="hand2",
        width=4,
        style='my.TButton',
        command=self.instructions_popup.destroy,
    )
    close_btn.place(anchor='n', relx=0.085, rely=0.85)  # Place close button at top-left corner
    # Optional: reset ref when closed
    self.instructions_popup.protocol("WM_DELETE_WINDOW", self.instructions_popup.destroy)

def load_PhotoImage(path, target_width=0):
    img = Image.open(path)
    if target_width == 0:
        return ImageTk.PhotoImage(img)
    aspect_ratio = img.height / img.width
    target_height = int(target_width * aspect_ratio)
    img = img.resize((target_width, target_height), Image.LANCZOS)
    return ImageTk.PhotoImage(img), target_height

def clear_canvas_show_results(self):
    '''Load the images page with the uploaded images and display them in a grid layout.'''
    for item in self.canvas.find_all():
        tags = self.canvas.gettags(item)
        if "bg" not in tags and "logo" not in tags and "bar" not in tags:
            self.canvas.delete(item)
            self.canvas.itemconfigure(self.progress_window, state='normal')
    self.canvas.update_idletasks()
    self.canvas.after(500, lambda: run_analysis_and_show_results(self))

def clear_canvas_show_home(self):
    '''Load the images page with the uploaded images and display them in a grid layout.'''
    for widget in self.canvas.winfo_children():
        widget.destroy()  # deleting widget
    self.canvas.destroy()
    del self.frame1
    self.__init__(self.root) # Reinitialize the app to reset the canvas and UI

def run_analysis_and_show_results(self):
    images_for_yolo = [np.array(d["original_pil"]) for d in self.loaded_images]
    images_for_yolo = [cv.cvtColor(img, cv.COLOR_RGB2BGR) for img in images_for_yolo]  # Convert to BGR for OpenCV
    stitched_img, heatmap_overlay, maxval, my_colormap = analyze_images(self, images_for_yolo, real_width_m=4)
    # Get canvas dimensions
    self.canvas.after(1000)  # Wait for canvas to update before getting dimensions

    bar_w, bar_h = 40, heatmap_overlay.shape[0]//2 # width, height in px
    grad = np.linspace(240, 0, bar_h, dtype=np.uint8).reshape(bar_h, 1)
    grad = np.repeat(grad, bar_w, axis=1)  # (256,50) column strip
    bar_img = cv.applyColorMap(grad, my_colormap)  # colourise with SAME LUT
    pad = 40
    bar_img = cv.copyMakeBorder(bar_img, 0, 0, 0, pad, cv.BORDER_CONSTANT, value=(255, 255, 255))
    min_val, max_val = 0, maxval  # Set min and max values for color bar
    cv.putText(bar_img, f"{max_val:.1f}", (bar_w + 5, 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    cv.putText(bar_img, f"{min_val:.1f}", (bar_w + 5, bar_h - 5),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
    #bar_rgb = cv.cvtColor(bar_img, cv.COLOR_BGR2RGB)  # BGR → RGB
    bar_pil = Image.fromarray(bar_img)  # numpy → PIL
    bar_tk = ImageTk.PhotoImage(bar_pil)  # PIL → Tk image

    stitched_pil = Image.fromarray(cv.cvtColor(stitched_img, cv.COLOR_BGR2RGB))
    heatmap_pil = Image.fromarray(cv.cvtColor(heatmap_overlay, cv.COLOR_BGR2RGB))
    self.canvas.itemconfigure(self.progress_window, state='hidden')
    if not hasattr(self, "frame1"):
        self.frame1 = tk.Frame(self.canvas)
        self.text1 = tk.Label(self.frame1, text="תמונה פנורמית של השדה", font=("Calibri", 22, "bold"))
        self.text1.pack(side="top", pady=(0, 2))
        self.label1 = tk.Label(self.frame1)
        self.label1.pack()
        self.button1 = CornButton(
            parent=self.canvas,
            text="לחצו להורדה",
            command=lambda: (
                                filename := filedialog.asksaveasfilename(
                                    defaultextension=".jpg",
                                    filetypes=[("JPEG files", "*.jpg")],
                                    initialfile="stitched_image.jpg"
                                )
                            ) and stitched_pil.save(filename, "JPEG", quality=100, optimize=False, subsampling=0),
            style='save.TButton',
            cursor="hand2",
            width=15,
            height=10
        )
        self.frame2 = tk.Frame(self.canvas)
        self.text2 = tk.Label(self.frame2, text="מפת חום - צפיפות למ''ר", font=("Calibri", 22, "bold"))
        self.label2 = tk.Label(self.frame2)
        self.text2.pack(side="top", pady=(0, 2))

        self.label33 = tk.Label(self.frame2, image=bar_tk)
        self.label33.image = bar_tk  # keep a reference so it isn’t garbage-collected
        self.label33.pack(side="right", padx=5, pady=5)
        self.label2.pack(side="top")
        self.button2 = CornButton(
            parent=self.canvas,
            text="לחצו להורדה",
            command=lambda: (
                                filename := filedialog.asksaveasfilename(
                                    defaultextension=".jpg",
                                    filetypes=[("JPEG files", "*.jpg")],
                                    initialfile="heatmap_image.jpg"
                                )
                            ) and heatmap_pil.save(filename, "JPEG", quality=100, optimize=False, subsampling=0),
            style='save.TButton',
            cursor="hand2",
            width=15,
            height=10
        )
        self.button3 = CornButton(
            parent=self.canvas,
            text="חזרה למסך הבית",
            command=lambda: clear_canvas_show_home(self),
            style='my.TButton',
            cursor="hand2",
            width=15,
            height=10
        )
        self.window1 = self.canvas.create_window(0, 0, window=self.frame1, anchor="n")
        self.window2 = self.canvas.create_window(0, 0, window=self.frame2, anchor="n")
        self.windowbutton1 = self.canvas.create_window(0, 0, window=self.button1.widget(), anchor="n")
        self.windowbutton2 = self.canvas.create_window(0, 0, window=self.button2.widget(), anchor="n")
        self.windowbutton3 = self.canvas.create_window(0, 0, window=self.button3.widget(), anchor="ne")
        self.canvas.itemconfigure(self.progress_window, state='hidden')

    def on_canvas_resize(event):
        if self.frame1.winfo_exists():
            w, h = event.width, event.height
            half_w = w // 2
            # Resize images to fit half the canvas
            aspect_ratio = stitched_pil.width / stitched_pil.height
            target_height = h - 400  # Leave space for other UI elements
            target_width = int(target_height * aspect_ratio)

            # Ensure width doesn't exceed half canvas
            if target_width > half_w:
                target_width = half_w - 200
                target_height = int(target_width / aspect_ratio)
            img1 = stitched_pil.resize((target_width, target_height), Image.LANCZOS)
            img2 = heatmap_pil.resize((target_width, target_height), Image.LANCZOS)
            tk_img1 = ImageTk.PhotoImage(img1)
            tk_img2 = ImageTk.PhotoImage(img2)
            self.label1.configure(image=tk_img1)
            self.label2.configure(image=tk_img2)
            self.label1.image = tk_img1
            self.label2.image = tk_img2
            self.canvas.coords(self.window1, w*0.25, 200)
            self.canvas.coords(self.window2, w*0.75, 200)
            self.canvas.coords(self.windowbutton1, self.canvas.coords(self.window1)[0],
                               200 + img1.height + self.button1.widget().winfo_height() - 5)
            self.canvas.coords(self.windowbutton2, self.canvas.coords(self.window2)[0],
                                200 + img2.height + self.button2.widget().winfo_height() - 5)
            self.canvas.coords(self.windowbutton3,
                               w * 0.96,
                               h * 0.06)

    # Bind the handler (if not already bound)
    self.canvas.bind("<Configure>", on_canvas_resize)
    # Force initial resize
    self.canvas.update_idletasks()
    on_canvas_resize(type('evt', (), {
        'width': self.canvas.winfo_width(), 'height': self.canvas.winfo_height()
    })())
