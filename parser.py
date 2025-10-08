import pandas as pd
import pymorphy3
import re
import json

INPUT_FILE = "full_dataset_clean.jsonl"    # входной JSONL (по одной записи в строке)
OUTPUT_FILE = "patents.json"     # обработанный JSON
morph = pymorphy3.MorphAnalyzer()

# ===============================
# Ключевые слова для классификации
# ===============================
TOPIC_KEYWORDS = {
    "Приборостроение и электроника": [
        # Русские
        "устройство", "датчик", "контроль", "электрон", "детект",
        "модуль", "генератор", "сигнал", "передача", "лазер", "изготов",
        "атор", "микросхем", "прибор", "плата", "контроллер", "усилител",
        "оптическ", "интерферометр", "волн", "схема", "интегральн",
        # Английские
        "device", "sensor", "control", "electronic", "detector", "module",
        "generator", "signal", "transmission", "laser", "manufacture",
        "chip", "board", "controller", "amplifier", "optical", "interferometer",
        "wave", "circuit", "integrated"
    ],
    "Физика и Ядерные технологии": [
        # Русские
        "ускорител", "реактор", "нейтрон", "ядро", "излучение",
        "детектор", "облучение", "радиация", "спектрометр", "пуч",
        "сверхпровод", "вакуум", "частиц", "магнит", "способ", "сцинтиллятор",
        "коллайдер", "фотон", "изотоп", "ядерн", "плазм", "энергетик",
        # Английские
        "accelerator", "reactor", "neutron", "nuclear", "radiation",
        "detector", "irradiation", "spectrometer", "beam", "superconduct",
        "vacuum", "particle", "magnet", "method", "scintillator",
        "collider", "photon", "isotope", "plasma", "fusion", "energy"
    ],
    "Материаловедение": [
        # Русские
        "наноматериал", "сплав", "структура", "термообработка",
        "композит", "порошок", "материал", "поверхность", "микроструктур",
        "кристалл", "тонкоплёночн", "термическ", "корроз", "механическ",
        # Английские
        "nanomaterial", "alloy", "structure", "heat", "composite", "powder",
        "material", "surface", "microstructure", "crystal", "thinfilm",
        "thermal", "corrosion", "mechanical"
    ],
    "Биомедицина": [
        # Русские
        "биологическ", "медицин", "терапия", "диагностика",
        "охлаждение", "образец", "пациент", "био", "ткань", "жив",
        "генетич", "заболев", "болез", "вакцин", "биоинженер",
        "клетк", "молекул", "иммун", "анализ", "диагностическ",
        # Английские
        "biological", "medical", "therapy", "diagnostic", "cooling",
        "sample", "patient", "bio", "tissue", "living", "genetic",
        "disease", "illness", "vaccine", "bioengineering", "cell",
        "molecule", "immune", "analysis", "diagnostics"
    ],
    "Информационные технологии": [
        # Русские
        "обработка", "алгоритм", "система", "программа",
        "вычислени", "моделирован", "управление", "данные",
        "информационн", "баз", "компьютер", "сеть", "облачн",
        "модульн", "нейросет", "распознаван", "анализ", "код",
        # Английские
        "processing", "algorithm", "system", "software", "computation",
        "modeling", "control", "data", "information", "database",
        "computer", "network", "cloud", "modular", "neural", "recognition",
        "analysis", "code", "machine", "learning", "artificial", "intelligence"
    ]
}

TOPIC_PRIORITY = [
    "Материаловедение",
    "Приборостроение и электроника",
    "Биомедицина",
    "Информационные технологии",
    "Физика и Ядерные технологии",
]

# ===============================
# Лингвистическая обработка
# ===============================
def clean_and_lemmatize(text: str) -> str:
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text)
    text = text.lower()
    words = text.split()
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)

# ===============================
# Классификация по темам
# ===============================
def classify_topic(text: str) -> str:
    matched = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched.append(topic)
                break
    if not matched:
        return "Не классифицировано"
    for t in TOPIC_PRIORITY:
        if t in matched:
            return t

# ===============================
# Основной блок
# ===============================
if __name__ == "__main__":
    # Чтение JSONL
    records = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)

    # Лемматизация названия и классификация
    df["Леммы_названия"] = df["name"].apply(clean_and_lemmatize)
    df["Леммы_текста"] = df["text_of_document"].apply(clean_and_lemmatize)
    df["Леммы_всё"] = df["Леммы_названия"] + " " + df["Леммы_текста"]
    df["Область"] = df["Леммы_всё"].apply(classify_topic)

    # Разбивка авторов
    df["Авторы"] = df["authors"].apply(
        lambda x: [a.strip() for a in str(x).split(",") if a.strip()] if x else ["Не указано"]
    )

    # Извлечение года
    df["Год"] = df["date"].str.extract(r"(\d{4})")
    df = df[
        df["Год"].notna() &
        (df["Год"] >= "2000") &
        (df["Область"] != "Не классифицировано")
    ]

    # Сохраняем в удобном формате
    df.drop(columns=["Леммы_названия", "Леммы_текста", "Леммы_всё"], inplace=True)
    df.to_json(OUTPUT_FILE, orient="records", force_ascii=False, indent=2)

    print(f"Обработано записей: {len(df)}")
    print(f"Сохранено в: {OUTPUT_FILE}")
