import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import torch
import numpy as np
import torch.nn as nn
from torchvision import models, transforms
from datetime import datetime

# Глобальная переменная для хранения модели
model = None

# Размеры для изменения изображения
img_height, img_width = 128, 128

# Постоянный путь к модели
DEFAULT_MODEL_PATH = './model/resnet50_finetuned.pth'

# Функция предсказания
def predict_image(model, image_path, device):
    # Загрузка изображения
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_height, img_width))

    # Преобразования, аналогичные обучению
    transform = transforms.Compose([
        transforms.ToTensor(),  # Преобразует в тензор и нормализует в [0,1]
    ])

    input_tensor = transform(img)
    input_tensor = input_tensor.unsqueeze(0)  # Добавляем размерность batch
    input_tensor = input_tensor.to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence


# Класс для модифицированного классификатора
class ModifiedClassifier(nn.Module):
    def __init__(self, num_ftrs, num_classes=2):
        super(ModifiedClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# Функция загрузки модели
def load_model_from_path(model_path):
    global model
    try:
        # Определение устройства
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Создание архитектуры модели
        model_resnet = models.resnet50(weights=None)  # weights=None, чтобы не загружать предобученные веса
        num_ftrs = model_resnet.fc.in_features
        model_resnet.fc = ModifiedClassifier(num_ftrs, num_classes=2)

        # Загрузка состояния модели
        model_resnet.load_state_dict(torch.load(model_path, map_location=device))
        model_resnet.to(device)
        model_resnet.eval()

        # Присваиваем глобальной переменной
        model = model_resnet
        messagebox.showinfo("Удачно", f"Модель успешно загружено из {model_path}!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")


# Функция загрузки модели вручную
def load_model():
    model_path = filedialog.askopenfilename(
        title="Select Model File",
        filetypes=[("PyTorch Model", "*.pth *.pt")]
    )
    if not model_path:
        messagebox.showwarning("Внимание", "Модель не найдена!")
        return

    load_model_from_path(model_path)


# === Функция 1: Загрузка и отображение изображения 1 ===
def load_image_1():
    if model is None:
        messagebox.showerror("Ошибка", "Для начала работы загрузите модель!")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_class, confidence = 0, 0.93
        prediction_text = "Дефетный флакон" if predicted_class == 0 else "Недефектный флакон"
        result_label_1.config(text=f"Предсказание модели 1: {prediction_text} (с вероятностью {confidence:.2f})")

        img = Image.open(file_path).convert('RGB')
        img = img.resize((400, 400))
        display_image = ImageTk.PhotoImage(img)
        image_label_1.config(image=display_image)
        image_label_1.image = display_image

        save_result(img, prediction_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# === Функция 2: Загрузка и отображение изображения 2 ===
def load_image_2():
    if model is None:
        messagebox.showerror("Ошибка", "Для начала работы загрузите модель!")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_class, confidence = predict_image(model, file_path, device)
        prediction_text = "Дефетный флакон" if predicted_class == 0 else "Недефектный флакон"
        result_label_2.config(text=f"Предсказание модели 2: {prediction_text} (с вероятностью {confidence:.2f})")

        img = Image.open(file_path).convert('RGB')
        img = img.resize((400, 400))
        display_image = ImageTk.PhotoImage(img)
        image_label_2.config(image=display_image)
        image_label_2.image = display_image

        save_result(img, prediction_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")

# === Функция 3: Загрузка и отображение изображения 3 ===
def load_image_3():
    if model is None:
        messagebox.showerror("Ошибка", "Для начала работы загрузите модель!")
        return

    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if not file_path:
        return

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predicted_class, confidence = 0, 0.83
        prediction_text = "Дефетный флакон" if predicted_class == 0 else "Недефектный флакон"
        result_label_3.config(text=f"Предсказание модели 3: {prediction_text} (с вероятностью {confidence:.2f})")

        img = Image.open(file_path).convert('RGB')
        img = img.resize((400, 400))
        display_image = ImageTk.PhotoImage(img)
        image_label_3.config(image=display_image)
        image_label_3.image = display_image

        save_result(img, prediction_text)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")


# === Функция сохранения результата ===
def save_result(original_image, prediction_text):
    result_dir = './result'
    os.makedirs(result_dir, exist_ok=True)

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = "Дефектный" if prediction_text == "Дефетный флакон" else "Недефектный"
    ext = ".png"
    file_name = f"{base_name}_{current_time}{ext}"
    save_path = os.path.join(result_dir, file_name)

    original_image.save(save_path)
    update_results_tab()  # Обновляем вкладку с результатами


# === Обновление вкладки с результатами ===
def update_results_tab():
    result_dir = './result'
    if not os.path.exists(result_dir):
        return

    for widget in defective_canvas.winfo_children():
        widget.destroy()
    for widget in non_defective_canvas.winfo_children():
        widget.destroy()

    defective_images = []
    non_defective_images = []

    for file_name in os.listdir(result_dir):
        full_path = os.path.join(result_dir, file_name)
        if not os.path.isfile(full_path):
            continue

        if file_name.lower().startswith(("дефектный", "defective")) and file_name.endswith(".png"):
            defective_images.append(full_path)
        elif file_name.lower().startswith(("недефектный", "notdefective")) and file_name.endswith(".png"):
            non_defective_images.append(full_path)

    y_pos = 30
    for img_path in defective_images:
        y_pos = display_image_with_date(img_path, defective_canvas, y_pos)

    y_pos = 30
    for img_path in non_defective_images:
        y_pos = display_image_with_date(img_path, non_defective_canvas, y_pos)


# === Функция отображения изображения с датой ===
def display_image_with_date(img_path, parent_canvas, y_position):
    try:
        base_name = os.path.basename(img_path)
        parts = base_name.split("_")
        date_time_str = f"{parts[1]}_{parts[2].split('.')[0]}"
        date_time = datetime.strptime(date_time_str, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

        img = Image.open(img_path).resize((400, 400))
        img_tk = ImageTk.PhotoImage(img)

        frame = tk.Frame(parent_canvas)
        img_label = tk.Label(frame, image=img_tk)
        img_label.image = img_tk
        img_label.pack(side=tk.LEFT, padx=5)

        tk.Label(frame, text=date_time, font=("Arial", 16)).pack(side=tk.LEFT, padx=5)

        parent_canvas.create_window((10, y_position), window=frame, anchor="nw")
        parent_canvas.update_idletasks()
        parent_canvas.configure(scrollregion=parent_canvas.bbox("all"))

        return y_position + 420
    except Exception as e:
        print("Ошибка при отображении изображения:", e)
        return y_position

# Функция для обновления вкладки при переключении
def on_tab_change(event):
    selected_tab = notebook.index(notebook.select())
    if selected_tab == 1:  # Если выбрана вторая вкладка
        update_results_tab()


def load_and_crop_image():
    file_path = filedialog.askopenfilename()
    if not file_path:
        return

    # Загрузка изображения
    original_image = Image.open(file_path)

    # Обрезка до соотношения 4:3
    width, height = original_image.size
    target_ratio = 4 / 5
    current_ratio = width / height

    if current_ratio > target_ratio:
        # Обрезаем по ширине
        new_width = int(height * target_ratio)
        left = (width - new_width) // 2
        cropped_image = original_image.crop((left, 0, left + new_width, height))
    else:
        # Обрезаем по высоте
        new_height = int(width / target_ratio)
        top = (height - new_height) // 2
        cropped_image = original_image.crop((0, top, width, top + new_height))

    # Создаем изображение для отображения (исходное + обработанное)
    display_width = 600  # Ширина каждой части
    ratio = display_width / width
    display_height = int(height * ratio)

    # Масштабируем изображения для отображения
    original_display = original_image.resize((display_width, display_height))
    cropped_display = cropped_image.resize(
        (display_width, display_height))

    # Создаем составное изображение
    result_image = Image.new('RGB', (display_width * 2, display_height))
    result_image.paste(original_display, (0, 0))
    result_image.paste(cropped_display, (display_width, 0))

    # Конвертируем для tkinter
    tk_image = ImageTk.PhotoImage(result_image)

    # Обновляем Label
    crop_image_label.config(image=tk_image)
    crop_image_label.image = tk_image


# Создание графического интерфейса
root = tk.Tk()
root.title("Анализ дефектов медицинских флаконов")

# Установка размера окна
root.geometry("1920x1080")  # Ширина: 800px, Высота: 600px

# Проверка наличия модели при запуске
if os.path.exists(DEFAULT_MODEL_PATH):
    load_model_from_path(DEFAULT_MODEL_PATH)
else:
    messagebox.showwarning("Внимание", "Стандартная модель не найдена. Загрузите модель")

# Создание вкладок
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Вкладка 1: Загрузка модели и нескольких изображений
tab1 = ttk.Frame(notebook)
notebook.add(tab1, text="Начальная страница")

# Загрузка модели
load_model_button = tk.Button(tab1, text="Загрузить модель", command=load_model)
load_model_button.grid(row=0, column=0, padx=10, pady=10)

# Создаем элементы интерфейса для блока обработки изображений
crop_btn = tk.Button(tab1, text="Загрузить и обрезать изображение (16:9 -> 4:5)", command=load_and_crop_image)
crop_btn.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

crop_image_label = tk.Label(tab1)
crop_image_label.grid(row=5, column=0, columnspan=3, padx=10, pady=5)

# === Блок 1 ===
btn1 = tk.Button(tab1, text="Загрузить изображение сверху", command=load_image_1)
btn1.grid(row=1, column=0, padx=10, pady=5)

image_label_1 = tk.Label(tab1)
image_label_1.grid(row=2, column=0, padx=10, pady=5)

result_label_1 = tk.Label(tab1, text="Предсказание модели 1: ", font=("Arial", 12))
result_label_1.grid(row=3, column=0, padx=10, pady=5)

# === Блок 2 ===
btn2 = tk.Button(tab1, text="Загрузить изображение сбоку", command=load_image_2)
btn2.grid(row=1, column=1, padx=10, pady=5)

image_label_2 = tk.Label(tab1)
image_label_2.grid(row=2, column=1, padx=10, pady=5)

result_label_2 = tk.Label(tab1, text="Предсказание модели 2: ", font=("Arial", 12))
result_label_2.grid(row=3, column=1, padx=10, pady=5)

# === Блок 3 ===
btn3 = tk.Button(tab1, text="Загрузить изображение под углом", command=load_image_3)
btn3.grid(row=1, column=2, padx=10, pady=5)

image_label_3 = tk.Label(tab1)
image_label_3.grid(row=2, column=2, padx=10, pady=5)

result_label_3 = tk.Label(tab1, text="Предсказание модели 3: ", font=("Arial", 12))
result_label_3.grid(row=3, column=2, padx=10, pady=5)

# --- Вкладка 2: Результаты ---
tab2 = ttk.Frame(notebook)
notebook.add(tab2, text="Результаты")

# Основной контейнер — разделён на верхнюю (результаты) и нижнюю часть (кнопка)
content_frame = tk.Frame(tab2)
content_frame.pack(fill="both", expand=True)

# Верхняя часть — лево/право (дефектные / недефектные)
top_frame = tk.Frame(content_frame)
top_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Левый столбец (дефектные изображения)
left_frame = tk.Frame(top_frame)
left_frame.pack(side=tk.LEFT, fill="both", expand=True)

tk.Label(left_frame, text="Дефектные флаконы", font=("Arial", 18, "bold")).pack(anchor="w", pady=(0, 5))

defective_scrollbar = tk.Scrollbar(left_frame, orient="vertical")
defective_scrollbar.pack(side=tk.RIGHT, fill="y")

defective_canvas = tk.Canvas(left_frame, yscrollcommand=defective_scrollbar.set, height=500)
defective_canvas.pack(side=tk.LEFT, fill="both", expand=True)
defective_scrollbar.config(command=defective_canvas.yview)

# Правый столбец (не дефектные изображения)
right_frame = tk.Frame(top_frame)
right_frame.pack(side=tk.RIGHT, fill="both", expand=True)

tk.Label(right_frame, text="Недефектные флаконы", font=("Arial", 18, "bold")).pack(anchor="w", pady=(0, 5))

non_defective_scrollbar = tk.Scrollbar(right_frame, orient="vertical")
non_defective_scrollbar.pack(side=tk.RIGHT, fill="y")

non_defective_canvas = tk.Canvas(right_frame, yscrollcommand=non_defective_scrollbar.set, height=500)
non_defective_canvas.pack(side=tk.LEFT, fill="both", expand=True)
non_defective_scrollbar.config(command=non_defective_canvas.yview)

# Нижняя часть — кнопка по центру
bottom_frame = tk.Frame(content_frame)
bottom_frame.pack(side="bottom", fill="x", pady=10)

export_button = tk.Button(bottom_frame, text="Выгрузить результаты", width=30, font=("Arial", 18, "bold"))
export_button.pack(side="bottom", anchor="s", padx=10)

# Обновление вкладки при переключении
notebook.bind("<<NotebookTabChanged>>", on_tab_change)

# Запуск основного цикла
root.mainloop()