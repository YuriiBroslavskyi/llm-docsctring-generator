import random

# Вкажи назву твого фінального файлу після мерджу
INPUT_FILE = "merged_20260406_214856.jsonl" 
TRAIN_FILE = "train_with_gemini_prompt.jsonl"
VAL_FILE = "val_with_gemini_prompt.jsonl"

def split_dataset(input_file, train_file, val_file, split_ratio=0.95):
    print("⏳ Читаємо та перемішуємо датасет...")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Обов'язково перемішуємо, щоб дані були різнорідними
    random.shuffle(lines)

    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    print("💾 Зберігаємо train.jsonl...")
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)

    print("💾 Зберігаємо val.jsonl...")
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)

    print("✅ Готово!")
    print(f"📚 Навчальна вибірка (Train): {len(train_lines)} записів")
    print(f"🎯 Валідаційна вибірка (Val): {len(val_lines)} записів")

if __name__ == "__main__":
    split_dataset(INPUT_FILE, TRAIN_FILE, VAL_FILE)