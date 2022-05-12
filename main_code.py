# #import the libraries
# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import matplotlib.pyplot as plt #Data Visualization

from tkinter import *
import train_model
import test_model
import time

import tkinter as tk
from tkinter import filedialog as tkfiledialdialog
from tkinter import ttk
from functools import partial
from PIL import Image, ImageTk

Font_tuple = ("Comic Sans MS", 20, "bold")


def openImage1(test_img):
    filename1 = test_img
    if not filename1:
        return

    else:
        image1 = Image.open(filename1)
        label4 = tk.Label(window, text='test image'.upper(), fg="#1a1aff",
                         bg='#e6e6ff', padx=3, pady=5)
        label4['font'] = ("Comic Sans MS", 13, "bold")
        label4.place(relx=0.4, rely=0.55, anchor=tk.CENTER)
        resize_img = image1.resize((250, 250))

        render1 = ImageTk.PhotoImage(resize_img)
        img = Label(window,image=render1)
        img.image = render1
        img.place(relx=0.4, rely=0.73, anchor=tk.CENTER)
        return True



def openImage2(train_img):
    filename = train_img
    if not filename:
        return
    else:
        image = Image.open(filename)
        label3 = tk.Label(window, text='matched image'.upper(), fg="#1a1aff",
                         bg='#e6e6ff', padx=3, pady=5)
        label3['font'] = ("Comic Sans MS", 13, "bold")
        label3.place(relx=0.68, rely=0.55, anchor=tk.CENTER)
        resize_img1 = image.resize((250, 250))
        render2 = ImageTk.PhotoImage(resize_img1)
        img = Label( window,image=render2)
        img.image = render2
        img.place(relx=0.7, rely=0.73, anchor=tk.CENTER)
        return True

def getfilepath_train():

    train_dir=tkfiledialdialog.askdirectory(parent=window,initialdir="/")
    if(train_dir!=""):
        train_model.train(train_dir)

def getfilepath_test():
    test_dir=tkfiledialdialog.askdirectory(parent=window,initialdir="/")
    if(test_dir!=""):
        c,total= test_model.test(test_dir, 'train_img_features_method2_copy.csv')
        accuracy=(c/total)*100
        output="Accuracy of the model is {}%".format(accuracy)
        T = tk.Text(height=3, width=50,fg='#7300e6',bg='#cce6ff',borderwidth=3, relief="ridge")

        label2 = tk.Label(window, text='Accuracy of the model'.upper(), fg="#1a1aff", bg='#e6e6ff', padx=15, pady=10)
        label2['font'] =("Comic Sans MS", 13, "bold")
        label2.place(relx=0.57, rely=0.27, anchor=tk.CENTER)

        T.place(relx=0.57, rely=0.35, anchor=tk.CENTER)
        T.insert(tk.END,output)
        #T.insert(tk.END," Accuracy of the model is 70.0%")


matched_img_path=""
def upload_image():
    global matched_img_path
    test_image=tkfiledialdialog.askopenfilename(initialdir='/',title='Upload Test Image',)

    if(test_image!=""):

        uploaded_status=openImage1(test_image)
        if(uploaded_status):
            status,matched_img_path= test_model.test(test_image, 'train_img_features_method2_copy.csv')
            #T1 = tk.Text(height=5, width=15,fg='#7300e6',bg='#cce6ff',borderwidth=3, relief="ridge")

            label5 = tk.Label(window, text='STATUS OF FACE RECOGNITION: '+status.upper(), fg="#1a1aff", bg='#e6e6ff', padx=15, pady=10)
            label5['font'] = ("Comic Sans MS", 13, "bold")
            label5.place(relx=0.57, rely=0.47, anchor=tk.CENTER)

            #T1.place(relx=0.57, rely=0.35, anchor=tk.CENTER)
            #T1.insert(tk.END,status)






def display_matched_img():
    global matched_img_path
    if(matched_img_path):
        train_dir = tkfiledialdialog.askdirectory(parent=window, title="select trainimages dir", initialdir="/")

        openImage2(train_dir + '/' + matched_img_path)
        matched_img_path=""
    else:
        return




window=tk.Tk()
window.title('Project Output')
window['bg']='#e6e6ff'
window.geometry('1000x1000')
label=tk.Label(window,text='Face Recognition Using Gradient Texture Features'.upper(),fg="#1a1aff",bg='#e6e6ff',padx=5, pady=10)
label['font']=Font_tuple
label.place(relx=0.5, rely=0.05, anchor=tk.CENTER)


button1=tk.Button(text="Train the model", fg='#9900ff', bg='#ffdd99', command=getfilepath_train, width=18, activeforeground='red', activebackground='#ccffee', borderwidth=3, relief="ridge", padx=5, pady=10)
button1['font']=("Courier", 12,"bold")
button1.place(relx=0.14, rely=0.17, anchor=tk.CENTER)


button2=tk.Button(text="Test the model",fg='#9900ff',bg='#ffdd99',command=getfilepath_test,width=18,activeforeground='red',activebackground='#ccffee',borderwidth=3, relief="ridge", padx=5, pady=10)
button2['font']=("Courier", 12,"bold")
button2.place(relx=0.38, rely=0.17, anchor=tk.CENTER)


button3=tk.Button(text="Choose the test image",fg='#9900ff',bg='#ffdd99',command=upload_image,width=20,activeforeground='red',activebackground='#ccffee',borderwidth=3, relief="ridge", padx=5, pady=10)
button3['font']=("Courier", 12,"bold")
button3.place(relx=0.61, rely=0.17, anchor=tk.CENTER)


button3=tk.Button(text="view Matched Image",fg='#9900ff',bg='#ffdd99',command=display_matched_img,width=18,activeforeground='red',activebackground='#ccffee',borderwidth=3, relief="ridge", padx=5, pady=10)
button3['font']=("Courier", 12,"bold")
button3.place(relx=0.85, rely=0.17, anchor=tk.CENTER)

# Separator object
separator = ttk.Separator(window, orient='horizontal')
separator.place(relx=0, rely=0.4, relwidth=1, relheight=0.001)








window.mainloop()