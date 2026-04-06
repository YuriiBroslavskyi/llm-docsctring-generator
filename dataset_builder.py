import ast
import os
import json
import copy

def is_valid_docstring(docstring):
    """
    Перевіряє, чи існує рядок документації і чи відповідає він Google Style.
    Відкидає короткі коментарі та залишає лише якісні описи.
    """
    if not docstring:
        return False
        
    # Мінімальний фільтр: шукаємо ключові слова з Google Style Guide
    # Можна додати 'Yields:', 'Raises:' за потребою
    has_args = "Args:" in docstring
    has_returns = "Returns:" in docstring
    
    # Беремо функцію, якщо є хоча б опис аргументів або результату
    return has_args or has_returns

def extract_and_clean_functions(filepath):
    """
    Парсить Python-файл, знаходить функції, витягує з них docstring,
    а потім очищає код функції від самої документації.
    """
    dataset_entries = []
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        # Будуємо Абстрактне Синтаксичне Дерево (AST)
        tree = ast.parse(source_code)
        
        # Проходимося по всіх вузлах дерева
        for node in ast.walk(tree):
            # Шукаємо звичайні та асинхронні функції
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                docstring = ast.get_docstring(node)
                
                if is_valid_docstring(docstring):
                    # Робимо копію вузла, щоб безпечно його модифікувати
                    clean_node = copy.deepcopy(node)
                    
                    # Видаляємо docstring з тіла функції
                    # В AST docstring завжди є першим елементом (Expr -> Constant)
                    if isinstance(clean_node.body[0], ast.Expr) and isinstance(clean_node.body[0].value, ast.Constant):
                        clean_node.body.pop(0)
                    
                    # Якщо функція містила ТІЛЬКИ docstring (тепер вона порожня), додаємо pass
                    if not clean_node.body:
                        clean_node.body.append(ast.Pass())
                        
                    # Перетворюємо очищене дерево назад у текст коду (Python 3.9+)
                    clean_code = ast.unparse(clean_node)
                    
                    # Формуємо пару для Instruction Dataset
                    entry = {
                        "instruction": "Generate a Google Style docstring for the following Python function.",
                        "input": clean_code,
                        "output": f'\"\"\"\n{docstring}\n\"\"\"'
                    }
                    dataset_entries.append(entry)
                    
    except SyntaxError:
        # Ігноруємо файли з синтаксичними помилками (іноді буває в опенсорсі)
        pass
    except Exception as e:
        print(f"Помилка при обробці {filepath}: {e}")
        
    return dataset_entries

def build_dataset(repo_path, output_file):
    """
    Рекурсивно обходить вказану директорію, шукає .py файли та зберігає результат у JSONL.
    """
    all_entries = []
    print(f"🚀 Починаємо сканування директорії: {repo_path}")
    
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith('.py'):
                filepath = os.path.join(root, file)
                entries = extract_and_clean_functions(filepath)
                all_entries.extend(entries)
    
    print(f"✅ Знайдено {len(all_entries)} функцій з ідеальною Google Style документацією.")
    
    # Зберігаємо у форматі JSONL (кожен рядок - окремий валідний JSON)
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"💾 Датасет успішно збережено у файл: {output_file}")

# ==========================================
# ЗАПУСК СКРИПТА
# ==========================================
if __name__ == "__main__":
    # 1. Шлях до папки з завантаженим опенсорс-кодом (наприклад, розпакований FastAPI)
    TARGET_REPO_DIR = "./target_repos" 
    
    # 2. Назва вихідного файлу датасету
    OUTPUT_DATASET = "docstrings_dataset.jsonl"
    
    # Якщо папки ще немає, створимо її, щоб скрипт не впав з помилкою
    if not os.path.exists(TARGET_REPO_DIR):
        os.makedirs(TARGET_REPO_DIR)
        print(f"Папку {TARGET_REPO_DIR} створено. Помістіть туди код бібліотек і запустіть скрипт знову.")
    else:
        build_dataset(TARGET_REPO_DIR, OUTPUT_DATASET)