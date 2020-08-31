from tkinter import Tk, Frame, Label, PhotoImage, Button
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from predecir import predict
filename=''
tipo=''
pos1=200
def ask():
    label3=Label(root,text="                                 ")
    label3.place(x=pos1,y=40)
    label5=Label(root,text="                                 ")
    label5.place(x=pos1,y=80)
    filename = askopenfilename() 
    label3=Label(root,text=filename)
    label3.place(x=pos1,y=40)
    tipo=predict(filename)
    label5=Label(root,text=tipo)
    label5.place(x=pos1,y=80)
root=Tk()
root.title("Trash sorter")
root.resizable(True,True)
root.geometry("650x500")
width=650
height=400
#frame=Frame(root, width=width, height=height)
#frame.pack()
Button(root, text="Choose Image", command=ask).place(x=550,y=40)
label1=Label(root,text='Trash sorter')
label1.place(x=325,y=10)
label2=Label(root,text='Selected image')
label2.place(x=40,y=40)

label4=Label(root,text="The image corresponds to")
label4.place(x=40,y=80)



#label2.image=photo
#label2.pack()
root.mainloop()

