import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageEnhance
import requests
import io
import os
from datetime import datetime
import threading
import cv2
import numpy as np
from fpdf import FPDF
import subprocess

# ------------ Helper functions ------------

def download_image(url):
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGBA")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def detect_art_crop_bounds(img, art_crop_img, threshold=0.6, debug=True):
    """Return (x1, y1, x2, y2) of best matched art crop area in img, or None if no match."""
    if img is None or art_crop_img is None:
        return None

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    art_crop_cv = cv2.cvtColor(np.array(art_crop_img), cv2.COLOR_RGBA2BGRA)

    h_img, w_img = img_cv.shape[:2]
    best_val = -1
    best_loc = None
    best_scale = 1.0
    best_size = (0, 0)

    for scale in [1.0, 0.95, 0.9, 0.85, 0.8, 1.05, 1.1]:
        new_w = int(art_crop_cv.shape[1] * scale)
        new_h = int(art_crop_cv.shape[0] * scale)
        if new_w >= w_img or new_h >= h_img:
            continue
        resized_template = cv2.resize(art_crop_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
        res = cv2.matchTemplate(img_cv, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_scale = scale
            best_size = (new_w, new_h)

    if best_val < threshold:
        if debug:
            print(f"[Warning] Art crop match failed (max match score: {best_val:.2f})")
        return None

    top_left = best_loc
    bottom_right = (top_left[0] + best_size[0], top_left[1] + best_size[1])
    if debug:
        print(f"Matched art crop at {top_left} with scale {best_scale:.2f} and score {best_val:.2f}")

    return (top_left[0], top_left[1], bottom_right[0], bottom_right[1])


def remove_art_using_crop(img, bounds):
    #bounds = detect_art_crop_bounds(img, art_crop_img)
    if bounds is None:
        return img

    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA)
    x1, y1, x2, y2 = bounds
    img_cv[y1:y2, x1:x2, 3] = 0  # Set alpha channel to 0 (transparent)

    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2RGBA))


def process_image(img, art_crop_img, style, dim_amount, remove_area, layout):
    img = img.convert("RGBA")
    w, h = img.size

    bounds = detect_art_crop_bounds(img, art_crop_img)
    if bounds is None:
        print("[Warning] Art not detected!")

    if remove_area and art_crop_img and not layout == "token":
        img = remove_art_using_crop(img, bounds)

    if style == "Grayscale":
        img = ImageOps.grayscale(img).convert("RGBA")

    elif style == "Dimmed":
        data = img.getdata()
        new_data = []
        dim_decimal = (min(((dim_amount)/100), 75))


        for i, item in enumerate(data):
            r, g, b, a = item
            x = i % w
            y = i // w
            in_art = False 
            if bounds:
                in_art = (bounds[0] <= x < bounds[2]) and (bounds[1] <= y < bounds[3])

            if (r, g, b) <= (80, 80, 80) and (not (in_art) or layout == "token"):
                new_data.append((r, g, b, a))  # Preserve black outside art
            else:
                r2 = int(r + (255 - r) * dim_decimal)
                g2 = int(g + (255 - g) * dim_decimal)
                b2 = int(b + (255 - b) * dim_decimal)
                new_data.append((r2, g2, b2, a))

        img.putdata(new_data)

    elif style == "Vibrant":
        enhancer = ImageEnhance.Color(img)
        vibrancy_decimal = (1+(dim_amount/100))
        img = enhancer.enhance(vibrancy_decimal)

    return img


def save_to_pdf(images, path):
    if not images:
        return
    pdf = FPDF(unit="pt", format="letter")  # 612x792 pts (8.5"x11")
    card_w, card_h = 180, 252  # MTG card dimensions in points (2.5" x 3.5")
    margin_x = (612 - (card_w * 3)) // 2  # Center horizontally
    margin_y = 15  # Fixed top margin, not vertically centered

    for i in range(0, len(images), 9):
        pdf.add_page()
        for j, img in enumerate(images[i:i+9]):
            x = margin_x + (j % 3) * card_w
            y = margin_y + (j // 3) * card_h
            tmp = f"tmp_{i+j}.png"
            img.resize((card_w, card_h)).save(tmp)
            pdf.image(tmp, x, y, card_w, card_h)
            os.remove(tmp)
    pdf.output(path)
    # Replace this with your actual PDF path variable
    pdf_path = path  # Example: "output_folder/filename.pdf"

    # Open the PDF automatically
    subprocess.call(["xdg-open", pdf_path])

# ------------ GUI ------------

class MTGApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MTG Card Downloader")
        self.geometry("750x550")

        self.create_widgets()

    def create_widgets(self):
        tk.Label(self, text="Enter cards (e.g. 2 Llanowar Elves):").pack(anchor="w")
        self.text = tk.Text(self, height=15)
        self.text.pack(fill="x", padx=5)

        style_frame = tk.Frame(self)
        style_frame.pack(padx=5, pady=5, fill="x")

        tk.Label(style_frame, text="Style:").pack(side="left")
        self.style_var = tk.StringVar(value="Normal")
        self.style_menu = ttk.Combobox(style_frame, textvariable=self.style_var, values=["Normal", "Grayscale", "Dimmed", "Vibrant"], state="readonly")
        self.style_menu.pack(side="left", padx=5)

        self.remove_var = tk.BooleanVar()
        tk.Checkbutton(style_frame, text="Remove Image", variable=self.remove_var).pack(side="left", padx=5)

        self.print_pdf_var = tk.BooleanVar(value =True)
        tk.Checkbutton(style_frame, text="Print to PDF", variable=self.print_pdf_var).pack(side="left", padx=5)

        self.download_images_var = tk.BooleanVar()
        tk.Checkbutton(style_frame, text="Download Images", variable=self.download_images_var).pack(side="left", padx=5)

        slider_frame = tk.Frame(self)
        slider_frame.pack(fill="x", padx=5)

        # Create IntVar to track the slider value
        self.dim_amount_var = tk.IntVar(value=50)

        # Label to show name and current value
        self.slider_label = ttk.Label(slider_frame, text=f"Dimming/Vibrancy Scalar -/+: {self.dim_amount_var.get()}%")
        self.slider_label.pack(pady=(10, 0), anchor="w")

        # Function to update label text when slider moves
        def update_label(val):
            self.slider_label.config(text=f"Dimming/Vibrancy Scalar -/+: {int(float(val))}%")

        # Create the slider
        ttk.Scale(
            slider_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.dim_amount_var,
            length=200,
            command=update_label
        ).pack(side="left", padx=5)

        filename_frame = tk.Frame(self)
        filename_frame.pack(fill="x", padx=5)
        tk.Label(filename_frame, text="Output file name:").pack(side="left")
        self.filename_entry = tk.Entry(filename_frame)
        self.filename_entry.pack(side="left", fill="x", expand=True, padx=5)

        folder_frame = tk.Frame(self)
        folder_frame.pack(fill="x", padx=5, pady=5)
        tk.Label(folder_frame, text="Output folder:").pack(side="left")
        self.folder_path_var = tk.StringVar()
        tk.Entry(folder_frame, textvariable=self.folder_path_var).pack(side="left", fill="x", expand=True, padx=5)
        tk.Button(folder_frame, text="Browse", command=self.browse_folder).pack(side="left")

        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x", padx=5, pady=10)

        self.download_btn = tk.Button(self, text="Download", command=self.start_download)
        self.download_btn.pack(pady=10)

    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_path_var.set(folder)

    def start_download(self):
        if not self.filename_entry.get().strip():
            messagebox.showerror("Error", "Please enter an output file name.")
            return
        if not self.folder_path_var.get():
            messagebox.showerror("Error", "Please select an output folder.")
            return
        card_lines = self.text.get("1.0", "end").strip().splitlines()
        if not card_lines or all(not line.strip() for line in card_lines):
            messagebox.showerror("Error", "Please enter at least one card.")
            return

        self.download_btn.config(state="disabled")
        self.progress["value"] = 0
        threading.Thread(target=self.download_cards, args=(card_lines,)).start()

    def download_cards(self, card_lines):
        # Parse cards input into list of (count, name)
        cards = []
        for line in card_lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 1:
                count = 1
                name = parts[0]
            else:
                try:
                    count = int(parts[0])
                    name = parts[1]
                except:
                    count = 1
                    name = line
            cards.append((count, name))

        # Deduplicate by name for image downloading
        unique_cards = {}
        for count, name in cards:
            unique_cards[name.lower()] = max(unique_cards.get(name.lower(), 0), count)

        total_images = sum(count for count, _ in cards)
        processed = 0
        style = self.style_var.get()
        dim_amount = self.dim_amount_var.get()
        remove_area = self.remove_var.get()
        download_images = self.download_images_var.get()
        output_folder = self.folder_path_var.get()
        base_filename = self.filename_entry.get().strip()
        layout = "normal"

        images_for_pdf = []

        # Create images output folder if needed
        images_folder = os.path.join(output_folder, "images", base_filename)
        if download_images:
            os.makedirs(images_folder, exist_ok=True)

        for card_name_lower, count in unique_cards.items():
            # Query Scryfall API for card data
            try:
                r = requests.get(f"https://api.scryfall.com/cards/named?exact={card_name_lower}")
                r.raise_for_status()
                card_data = r.json()
            except Exception as e:
                messagebox.showerror("Error", f"Card not found or API error: {card_name_lower}")
                #self.download_btn.config(state="normal")
                
                self.after(0, lambda: self.download_btn.config(state="normal"))
                return

            if "image_uris" in card_data and "large" in card_data["image_uris"]:
                img_url = card_data["image_uris"]["large"]
                art_crop_url = card_data["image_uris"].get("art_crop", None)
            elif "card_faces" in card_data and card_data["card_faces"]:
                img_url = card_data["card_faces"][0]["image_uris"]["large"]
                art_crop_url = card_data["card_faces"][0]["image_uris"].get("art_crop", None)
            else:
                print(f"Warning: No image found for {card_data.get('name', 'Unknown card')}")
                continue  # or handle gracefully

            if "layout" in card_data:
                layout = card_data["layout"]

            img = download_image(img_url)
            #art_crop_img = download_image(art_crop_url) if (remove_area and art_crop_url) else None
            correct_style = False
            correct_style = remove_area or (style == "Dimmed")
            art_crop_img = download_image(art_crop_url) if (correct_style) else None

            if img is None:
                messagebox.showerror("Error", f"Failed to download image for: {card_name_lower}")
                # self.download_btn.config(state="normal")
                
                self.after(0, lambda: self.download_btn.config(state="normal"))
                return

            # Process image once per unique card
            processed_img = process_image(img, art_crop_img, style, dim_amount, remove_area, layout)

            # Save image if requested
            if download_images:
                safe_name = card_name_lower.replace(" ", "_").replace("/", "_")
                for i in range(count):
                    save_path = os.path.join(images_folder, f"{safe_name}_{i+1}.png")
                    processed_img.save(save_path)

            # Add processed image to pdf list count times
            images_for_pdf.extend([processed_img.copy()] * count)

            processed += count
            self.progress["value"] = (processed / total_images) * 100

        # Save PDF if requested
        if self.print_pdf_var.get():
            pdf_path = os.path.join(output_folder, f"{base_filename}.pdf")
            save_to_pdf(images_for_pdf, pdf_path)

        self.progress["value"] = 100
        self.after(0, lambda: self.download_btn.config(state="normal"))

if __name__ == "__main__":
    app = MTGApp()
    app.mainloop()
