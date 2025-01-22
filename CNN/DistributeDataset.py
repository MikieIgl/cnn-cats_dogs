import os
import shutil
import random
import zipfile


# Процедура для распределения изображений
def distribute_images_from_folders(*, source_folder: str, dataset_folder: str) -> None:
    """
    Распределяет изображения кошек и собак из заданной папки по соответствующим
    директориям для обучения, валидации и тестирования.

    Параметры:
    ----------
    source_folder : str
        Путь к папке, содержащей подкаталоги с изображениями кошек ('Cat')
        и собак ('Dog').

    dataset_folder : str
        Путь к папке, в которую будут перемещены изображения для обучения,
        валидации и тестирования.

    Возвращает:
    ----------
    None
        Функция не возвращает значений, но перемещает изображения в
        соответствующие директории.
    """
    # Создание необходимых директорий
    os.makedirs(os.path.join(dataset_folder, 'training', 'train', 'Cats'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'training', 'train', 'Dogs'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'training', 'val', 'Cats'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'training', 'val', 'Dogs'), exist_ok=True)

    os.makedirs(os.path.join(dataset_folder, 'testing', 'Cats'), exist_ok=True)
    os.makedirs(os.path.join(dataset_folder, 'testing', 'Dogs'), exist_ok=True)

    # Получаем списки изображений
    cat_images = os.listdir(os.path.join(source_folder, 'Cat'))
    dog_images = os.listdir(os.path.join(source_folder, 'Dog'))

    # Перемешивание изображений
    random.shuffle(cat_images)
    random.shuffle(dog_images)

    t, v = 0.9, 0.1                         # Соотношение тренировочных и валидационных данных
    tr, te = 0.8, 0.2                       # Соотношение тренировочных и тестовых данных
    cat_images_total = len(cat_images)      # Количество изображений кошек
    dog_images_total = len(dog_images)      # Количество изображений собак

    # Определение количеств изображений
    testing_cat_img, testing_dog_img = int(cat_images_total * te), int(dog_images_total * te)
    training_cat_img, training_dog_img = int(cat_images_total * tr), int(dog_images_total * tr)
    train_cat_img, train_dog_img = int(training_cat_img * t), int(training_dog_img * t)
    val_cat_img, val_dog_img = int(training_cat_img * v), int(training_dog_img * v)

    # Перемещение изображений для train
    for img in cat_images[:train_cat_img]:
        shutil.move(os.path.join(source_folder, 'Cat', img),
                    os.path.join(dataset_folder, 'training', 'train', 'Cats', img))
    for img in dog_images[:train_dog_img]:
        shutil.move(os.path.join(source_folder, 'Dog', img),
                    os.path.join(dataset_folder, 'training', 'train', 'Dogs', img))

    # Перемещение изображений для val
    for img in cat_images[train_cat_img:train_cat_img + val_cat_img]:
        shutil.move(os.path.join(source_folder, 'Cat', img),
                    os.path.join(dataset_folder, 'training', 'val', 'Cats', img))
    for img in dog_images[train_dog_img:train_dog_img + val_dog_img]:
        shutil.move(os.path.join(source_folder, 'Dog', img),
                    os.path.join(dataset_folder, 'training', 'val', 'Dogs', img))

    # Перемещение оставшихся изображений для testing
    for img in cat_images[training_cat_img:training_cat_img + testing_cat_img]:
        shutil.move(os.path.join(source_folder, 'Cat', img),
                    os.path.join(dataset_folder, 'testing', 'Cats', img))
    for img in dog_images[training_cat_img:training_cat_img + testing_dog_img]:
        shutil.move(os.path.join(source_folder, 'Dog', img),
                    os.path.join(dataset_folder, 'testing', 'Dogs', img))


# Извлечение ZIP-архива
zip_path = r"D:\CNN_Cats_and_Dogs\archive.zip"
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(os.path.dirname(zip_path))

images_folder = r"D:\CNN_Cats_and_Dogs\PetImages"           # Путь к папке с изображениями после разархивирования
new_folder = r"D:\CNN_Cats_and_Dogs\cats_dogs_dataset"      # Путь к папке для распределения изображений

# Вызов процедуры для распределения изображений
distribute_images_from_folders(source_folder=images_folder, dataset_folder=new_folder)
