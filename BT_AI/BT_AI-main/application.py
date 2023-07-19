from tkinter import *
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image, ImageGrab
import time
import numpy as np
import tensorflow as tf

#Root setup
root = Tk()
root.wm_title("Tumor classification")
root.resizable(width = True, height = True)

#Importing model
model = tf.keras.models.load_model('C:/Users/tomke/Downloads/model/bt_ai_EXP3-91acc-04lss-08-04-23.h5', compile=False)

#Canvas setup
canvas = Canvas(root, width=1920, height=1080)
canvas.grid(row=0, column=1, padx=10, pady=5)

top_frame = Frame(canvas, width=1820, height=50, bg='grey')
canvas.create_window(0, 0, anchor='nw', window=top_frame, width=1920)

right_frame = Canvas(root, width=650, height=400, bg='black')
canvas.create_window(300, 50, anchor='nw', window=right_frame)

warning_label = Label(right_frame, width=170, height=50, text=f"To insert an image, click the 'Open image' button on the top left of the screen", bg="black", fg="white")
warning_label.grid(row=1, column=0, padx=5, pady=5, sticky="NW")

def info():
    window = Toplevel()
    window.wm_title("Info")
    label = Label(window, text="CNN model statistics:\n93% acc\n0.38 loss")
    label.pack()

def open_img():
    file = filedialog.askopenfilename()
    image = Image.open(file)
    photo = ImageTk.PhotoImage(image)
    
    #inits for ml model
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    #Initial black screen
    img_label = Label(right_frame, width=1280, height=720, bg='black')
    img_label.image = photo  # assign the photo to the label's instance variable
    img_label.config(image=photo)
    img_label.grid(row=1, column=0, padx=5, pady=5)
    
    #Filename
    filename = file.split("/")[-1]
    name_label = Label(right_frame, text=f"{filename}", bg="black", fg="white")
    name_label.grid(row=0, column=0, padx=5, pady=5, sticky="NE")

    #height and width
    heightweight_label = Label(right_frame, text=f"VIEW SIZE: (650, 400)\nIMG SIZE:{image.size}", bg="black", fg="white")
    heightweight_label.grid(row=1, column=0, padx=5, pady=5, sticky="NE")

    #Classification types
    classification_label = []
    for i in range(4):
        class_label = Label(right_frame, bg="black", fg="White")
        classification_label.append(class_label)
        class_label.grid(row=i+1, column=0, padx=2, pady=2, sticky="SW")

    predictions = model.predict(image_array)[0]
    for i in range(4):
        label = ""
        if i == 0:
            label = "Glioma confidence"
        elif i == 1:
            label = "Meningioma confidence"
        elif i == 2:
            label = "No tumor confidence"
        else:
            label = "Pituitary confidence"
        percent = predictions[i]
        classification_label[i].configure(text=f"{label}: {percent:.2%}")

    #misc
    left_frame = Frame(canvas, width=200, height=400, bg='grey')
    canvas.create_window(0, 50, anchor='nw', window=left_frame)

    editor_label = Label(right_frame, text=f"Analysed by BT_AI", bg="black", fg="white")
    editor_label.grid(row=0, column=0, padx=5, pady=5, sticky="NW")
    
    date_time_label = Label(right_frame, text=f"{time.strftime('%m/%d/%Y %H:%M:%S')}", bg="black", fg="white")
    date_time_label.grid(row=1, column=0, padx=5, pady=5, sticky="NW")

    augment_title = Label(left_frame, text=f"Augmentation settings", bg="grey", fg="black")
    augment_title.grid(row=1, column=0, padx=5, pady=5, sticky="NE")

    #Image augmentation functions

    def reset_img():
        photo = ImageTk.PhotoImage(image)
        img_label.config(image=photo)
        img_label.image = photo
    
    reset_button = Button(left_frame, text="Reset", command=reset_img)
    reset_button.grid(row=2, column=0, padx=5, pady=5)

    def flip_image_horizontally():
        imag_mod = image.transpose(method=Image.FLIP_LEFT_RIGHT)
        photo = ImageTk.PhotoImage(imag_mod)
        img_label.config(image=photo)
        img_label.image = photo

    flip_button = Button(left_frame, text="Flip horizontally", command=flip_image_horizontally)
    flip_button.grid(row=3, column=0, padx=5, pady=5)

    def flip_image_vertical():
        #global image
        imag_mod = image.transpose(method=Image.FLIP_TOP_BOTTOM)
        photo = ImageTk.PhotoImage(imag_mod)
        img_label.config(image=photo)
        img_label.image = photo

    flip_horiz_button = Button(left_frame, text="Flip Vertically", command=flip_image_vertical)
    flip_horiz_button.grid(row=4, column=0, padx=5, pady=5)

    
#Saving an image
def save():
    file_path = filedialog.asksaveasfilename(defaultextension='.jpg')
    
    x = root.winfo_rootx() + right_frame.winfo_x()
    y = root.winfo_rooty() + right_frame.winfo_y()
    
    width = 1290
    height = 900

    right = x + width
    bottom = y + height
    
    img = ImageGrab.grab(bbox=(x, y, right, bottom))
    img.save(file_path)
        


#Extra buttons for top frame
Button(top_frame, text='Open image', command = open_img).pack(side=LEFT, padx=5, pady=5)
Button(top_frame, text='About model', command=info).pack(side=LEFT, padx=5, pady=5)
Button(top_frame, text='Export as PNG/JPG', command=save).pack(side=TOP, padx=5, pady=5)

#Initial warning message
messagebox.showinfo("Information", "Before using this software, please be aware that any predictions made in this program must only be used as a reference as AI predictions may not be fully accurate. For more information, click 'About model' for more details")
root.mainloop()