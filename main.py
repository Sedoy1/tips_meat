import tkinter as tk
from tkinter import filedialog
import cv2
import torch
from PIL import Image, ImageTk
from torchvision import transforms, models


class ImageApp:
    def __init__(self, master, ask_model):
        self.img_label = None
        self.master = master
        self.master.title("Image App")

        self.img = None
        self.ask_model = ask_model

        # Кнопки инит
        self.choose_button = tk.Button(master, text="Выбрать изображение", command=self.choose_image)
        self.choose_button.pack(pady=10)

        self.capture_button = tk.Button(master, text="Открыть камеру и сфотографировать", command=self.capture_image)
        self.capture_button.pack(pady=10)

        self.recognize_button = tk.Button(master, text="Распознать", command=self.recognize_image)
        self.recognize_button.pack(pady=10)

        self.repeat_button = tk.Button(master, text="Повтор", command=self.repeat)
        self.repeat_button.pack(pady=10)

        # Вывод результата распознавания
        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

    def choose_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.img = cv2.imread(file_path)
            self.img = cv2.resize(self.img, (400, 400))
            self.display_image()

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()

        if ret:
            self.img = cv2.resize(frame, (400, 400))
            self.display_image()

    def recognize_image(self):
        if self.img is not None:
            result = ask_image(self.img, self.ask_model)
            self.result_label.config(text=f"Результат: {result}")
        else:
            self.result_label.config(text="Выберите или сфотографируйте изображение")

    def repeat(self):
        # Очистка изображения и результата
        self.img_label.destroy()
        self.img = None
        self.result_label.config(text="")
        self.choose_button.config(state=tk.NORMAL)
        self.capture_button.config(state=tk.NORMAL)

    def display_image(self):
        img_rgb = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        img_tk = ImageTk.PhotoImage(Image.fromarray(img_rgb))

        # Отображение изображения
        self.img_label = tk.Label(self.master, image=img_tk)
        self.img_label.image = img_tk
        self.img_label.pack()


        self.choose_button.config(state=tk.DISABLED)
        self.capture_button.config(state=tk.DISABLED)


# Распознавание изображения с использованием модели PyTorch
def ask_image(img, model):
    img = torch.tensor(img).to(torch.float).transpose(2, 0) / 256
    img = img[[2, 1, 0], :, :]
    with torch.no_grad():
        predict = model(img.unsqueeze(0))
    predict = torch.nn.functional.softmax(predict)
    print(predict)
    return "ROTTEN" if predict[0][0] > predict[0][1] else "FRESH"


new_model = models.resnet50(pretrained=True)
num_ftrs = new_model.fc.in_features
new_model.fc = torch.nn.Linear(num_ftrs, 2)
new_model.load_state_dict(torch.load("tips_meat/main/weights.pt"))

# Запуск
root = tk.Tk()
app = ImageApp(root, new_model)
root.mainloop()
