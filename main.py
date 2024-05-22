from customtkinter import CTk, CTkButton, CTkLabel, set_appearance_mode
from analyze_face_properties_live import main
from upload_image import main_func
from open_sigmoid_site import open_website
import tkinter as tk
import random


class Particle:
    def __init__(self, canvas, x, y, size=8, color='light blue'):
        self.canvas = canvas
        self.size = size
        self.id = canvas.create_oval(x, y, x + size, y + size, fill=color)
        self.dx = random.choice([-1, 1]) * random.uniform(1, 3.5)
        self.dy = random.choice([-1, 1]) * random.uniform(1, 3.5)

    def move(self):
        coords = self.canvas.coords(self.id)
        if coords[0] <= 0 or coords[2] >= self.canvas.winfo_width():
            self.dx = -self.dx
        if coords[1] <= 0 or coords[3] >= self.canvas.winfo_height():
            self.dy = -self.dy
        self.canvas.move(self.id, self.dx, self.dy)

class ParticleSimulation:
    def __init__(self, root, width=2000, height=1000, particle_count=500):
        self.root = root
        self.canvas = tk.Canvas(root, width=width, height=height, bg='black')
        self.canvas.pack()
        self.particles = [Particle(self.canvas, random.uniform(0, width), random.uniform(0, height)) for _ in range(particle_count)]
        self.animate()

    def animate(self):
        for particle in self.particles:
            particle.move()
        self.root.after(20, self.animate)


app = CTk()
app.geometry("700x700")
set_appearance_mode("dark")
simulation = ParticleSimulation(app)


btn_live_rec = CTkButton(master=app,
                         text="Live Face Analyze",
                         corner_radius=62,
                         fg_color="transparent",
                         hover_color="#FFA500",
                         border_color="lime green",
                         border_width=5,
                         font=("Arial", 30, "bold"),
                         cursor="hand2",
                         command=main)

label_or = CTkLabel(master=app,
                    text="or",
                    font=("Georgia", 30, "bold"))

btn_upload_img = CTkButton(master=app,
                           text="Upload Image",
                           corner_radius=62,
                           fg_color="transparent",
                           hover_color="#FFA500",
                           border_color="lime green",
                           border_width=5,
                           font=("Arial", 30, "bold"),
                           cursor="hand2",
                           command=main_func)

sigmoid_link = CTkButton(master=app,
                         text="Check Sigmoid Site",
                         font=("Georgia", 20, "bold"),
                         corner_radius=52,
                         fg_color="transparent",
                         hover_color="lime green",
                         cursor="hand2",
                         command=open_website)

btn_live_rec.place(relx=0.5, rely=0.5, anchor="center")
btn_upload_img.place(relx=0.5, rely=0.64, anchor="center")
label_or.place(relx=0.5, rely=0.57, anchor="center")
sigmoid_link.place(relx=0.84, rely=0.96, anchor="center")



app.mainloop()
