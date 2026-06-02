
import cv2
import os
import sys

TARGET_SIZE = 640

INPUT_DIR  = "input_photos"
OUTPUT_DIR = "cropped_photos"

SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def crop_to_square_center(img):

    h, w = img.shape[:2]
    side = min(h, w)

    y1 = (h - side) // 2
    x1 = (w - side) // 2
    y2 = y1 + side
    x2 = x1 + side

    return img[y1:y2, x1:x2]


def process_image(input_path: str, output_path: str) -> bool:

    img = cv2.imread(input_path)
    if img is None:
        print(f"  ПОМИЛКА: не вдалося відкрити '{input_path}'")
        return False

    h, w = img.shape[:2]

    square = crop_to_square_center(img)

    if square.shape[0] != TARGET_SIZE:
        final = cv2.resize(square, (TARGET_SIZE, TARGET_SIZE),
                           interpolation=cv2.INTER_AREA)
    else:
        final = square

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, final)

    print(f"  OK  {os.path.basename(input_path):30s}  "
          f"{w}x{h} → {TARGET_SIZE}x{TARGET_SIZE}")
    return True


def process_single(input_path: str):
    name, ext = os.path.splitext(os.path.basename(input_path))
    if ext.lower() not in SUPPORTED:
        print(f"Непідтримуваний формат: {ext}")
        sys.exit(1)

    output_path = os.path.join(OUTPUT_DIR, f"{name}_640{ext}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\nОбробка одного файлу: {input_path}")
    print("-" * 55)
    ok = process_image(input_path, output_path)

    if ok:
        print(f"\nЗбережено: {output_path}")


def process_folder(input_dir: str, output_dir: str):
    """Обробка всіх фото у папці."""
    if not os.path.exists(input_dir):
        print(f"ПОМИЛКА: папка '{input_dir}' не існує.")
        print(f"Створіть папку '{input_dir}' і покладіть туди фото.")
        sys.exit(1)

    files = [
        f for f in sorted(os.listdir(input_dir))
        if os.path.splitext(f)[1].lower() in SUPPORTED
    ]

    if not files:
        print(f"У папці '{input_dir}' не знайдено жодного фото.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\nЗнайдено {len(files)} фото у '{input_dir}'")
    print(f"Результат буде збережено у '{output_dir}'")
    print("-" * 55)

    success = 0
    for filename in files:
        input_path  = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        if process_image(input_path, output_path):
            success += 1

    print("-" * 55)
    print(f"Готово: {success}/{len(files)} фото оброблено успішно")
    print(f"Збережено у папці: '{output_dir}'\n")


# --- Точка входу ---
if __name__ == "__main__":
    args = sys.argv[1:]

    if len(args) == 0:

        process_folder(INPUT_DIR, OUTPUT_DIR)

    elif len(args) == 1:

        process_single(args[0])

    elif len(args) == 2:

        process_folder(args[0], args[1])

    else:
        print("Використання:")
        print("  python crop_photos.py                        # папка input_photos")
        print("  python crop_photos.py фото.jpg               # одне фото")
        print("  python crop_photos.py вхідна_папка вихідна_папка")
        sys.exit(1)
