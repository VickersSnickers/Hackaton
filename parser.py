import pandas as pd
import pymorphy3
import re

URL = "https://oliis.jinr.ru/index.php/patentovanie-2/8-russian/25-dejstvuyushchie-patenty-oiyai"
morph = pymorphy3.MorphAnalyzer()

def clean_and_lemmatize(text):
    text = re.sub(r"<.*?>", " ", str(text))
    text = re.sub(r"[^а-яА-Яa-zA-Z0-9\s]", " ", text)
    text = text.lower()
    words = text.split()
    lemmas = [morph.parse(w)[0].normal_form for w in words]
    return " ".join(lemmas)

TOPIC_KEYWORDS = {
    "Приборостроение и электроника": [
        "устройство", "датчик", "контроль", "электрон",  "детект"
        "модуль", "генератор", "сигнал", "передача", "лазер", "изготов", "атор"
    ],
    "Физика и Ядерные технологии": [
        "ускорител", "реактор", "нейтрон", "ядро", "излучение",
        "детектор", "облучение", "радиация", "спектрометр", "пуч",
        "сверхпровод", "вакуум", "частиц", "магнит", "способ", "сцинтиллятор"
    ],
    "Материаловедение": [
        "наноматериал", "сплав", "структура", "термообработка",
        "композит", "порошок", "материал", "поверхность"
    ],
    "Биомедицина": [
        "биологическ", "медицин", "терапия", "диагностика", 
        "охлаждение", "образец", "пациент", "био", "ткань", "жив",
        "генетич", "заболев", "болез"
    ],
    "Информационные технологии": [
        "обработка", "алгоритм", "система", "программа", 
        "вычислени", "моделирован", "управление", "данные"
    ]
}

TOPIC_PRIORITY = [
    "Материаловедение",
    "Приборостроение и электроника",
    "Биомедицина",
    "Информационные технологии",
    "Физика и Ядерные технологии",
]

def classify_topic_with_priority(text):
    matched_topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                matched_topics.append(topic)
                break
    if not matched_topics:
        return "Не классифицировано"
    for priority_topic in TOPIC_PRIORITY:
        if priority_topic in matched_topics:
            return priority_topic


if __name__ == "__main__":
    tables = pd.read_html(URL)
    df = tables[1]
    df.columns = ["№ п/п", "Номер патента", "Приоритет", "Название", "Публикации", "Авторы"]
    df = df.drop(columns=["№ п/п"])
    df = df[df["Номер патента"] != "Номер патента"]
    df["Авторы"] = df["Авторы"].apply(lambda x: [a.strip() for a in str(x).split(",")])
    df = df.drop(index=0)

    df["Чистое_название"] = df["Название"].apply(clean_and_lemmatize)
    df["Область"] = df["Чистое_название"].apply(classify_topic_with_priority)

    df.drop(columns=["Чистое_название"], inplace=True)
    df.to_json("patents.json", orient="records", force_ascii=False, indent=2)
    print("Количество записей:", len(df))