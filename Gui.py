import tkinter as tk
from tkinter import filedialog
from Manager import *
from PIL.ImageTk import PhotoImage
from tkVideoPlayer import TkinterVideo
import threading
import time as tm
from tkinter import messagebox
from tkinter import ttk


path_string = ""
seconds = 1
emotion = None




def play():
    """
    Launches the video in a new window and starts an analysis thread
    """
    if path_string == "":
        messagebox.showerror("empty path", "you should upload video")

    else:
        videoframe = tk.Toplevel()
        videoframe.geometry("400x400")
        button3 = tk.Button(videoframe, text="Exit", width=60,command=lambda: Exit(videoframe),
                            background='#5DF7FF')
        button3.pack()
        videoplayer = TkinterVideo(videoframe, scaled=True)
        videoplayer.load(path_string)
        videoplayer.pack(expand=True, fill="both")
        videoplayer.play()
        threading.Thread(target=lambda:analyze(videoframe)).start()
        videoframe.mainloop()


def upload_video(label):
    """
    Allows the user to select a video file to analyze and updates the label to show the path of the selected file
    """
    global path_string
    global emotion
    print('upload_video')
    folder_selected = filedialog.askopenfile()
    if folder_selected is None:
        return
    str =  folder_selected.name
    extension = str[-3:]
    if extension == 'mp4':
        path_string =  folder_selected.name
        print(path_string)
        threading.Thread(target=lambda: progressbar()).start()

        label.configure(text=f'Video path: {folder_selected.name}')

    else:
        messagebox.showerror("error file", "you should load mp4 file")




def analyze(videoframe):
    """
    Displays the predicted emotion on the video frame
    """
    emotion_label = tk.Label(videoframe, height=1, width=60, background='#5DF7FF')
    emotion_label.pack(pady=3)
    for i in emotion:
        emotion_label.configure(text=i,font=25)
        videoframe.update()
        tm.sleep(seconds)
    print(emotion)
    emotion_label.destroy()


def progressbar():
    global emotion
    pb = ttk.Progressbar(root, orient='horizontal', mode='indeterminate', length=280)
    pb.pack(pady=3)
    pb.start()
    while emotion is None:
        emotion = run(path_string, seconds)
        root.update()
    pb.destroy()
def Exit(frame):
    frame.destroy()



root = tk.Tk()

# Adjust size
root.geometry("400x400")

# Add image file
bg = PhotoImage(file=r"C:\Users\eyal_a\Downloads\_110124398_mediaitem110121043.jpg", size=(400, 400))

# Show image using label
label1 = tk.Label(root, image=bg, width=400, height=400)
label1.place(x=0, y=0)

# Add buttons
label1 = tk.Label(root)
label1.pack(pady=100)


button1 = tk.Button(root, text="play", width=30, command=lambda: threading.Thread(target=play()).start(), background='#5DF7FF')
button1.pack()
label_path = tk.Label(root, height=1, background='#5DF7FF')
button2 = tk.Button(root, text="upload video", width=30, command=lambda: upload_video(label_path), background='#5DF7FF')
button2.pack(pady=3)

label_path.pack()
button4 = tk.Button(root, text="Exit", width=30, command=lambda: Exit(root), background='#5DF7FF')
button4.pack(pady=3)


# videoplayer.play()
# Execute tkinter
root.mainloop()

