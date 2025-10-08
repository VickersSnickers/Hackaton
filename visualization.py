import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ===============================
# Настройки
# ===============================
INPUT_FILE = "patents.json"
OUTPUT_DIR = Path("./images")
OUTPUT_DIR.mkdir(exist_ok=True)

sns.set(style="whitegrid", font_scale=1.1)
plt.rcParams["axes.unicode_minus"] = False

# ===============================
# Загрузка данных
# ===============================
df = pd.read_json(INPUT_FILE)

# Фильтруем пустые и неклассифицированные
df = df[(df["Год"].notna()) & (df["Область"] != "Не классифицировано")]

# Взрываем список авторов
df_exploded = df.explode("Авторы")
df_exploded = df_exploded[
    df_exploded["Авторы"].notna() &
    (df_exploded["Авторы"].str.strip() != "") &
    (df_exploded["Авторы"].str.lower() != "не указано")
]

# ===============================
# Подсчёты
# ===============================
year_counts = df["Год"].value_counts().sort_index()
author_counts = df_exploded["Авторы"].value_counts()
area_counts = df["Область"].value_counts()

heatmap_data = (
    df_exploded.groupby(["Авторы", "Год"])
    .size()
    .unstack(fill_value=0)
)

year_area_counts = (
    df.groupby(["Год", "Область"])
    .size()
    .unstack(fill_value=0)
)

# ===============================
# Сохранение графиков
# ===============================
def save_plot(fig, name):
    fig.savefig(OUTPUT_DIR / f"{name}.png", bbox_inches="tight")
    plt.close(fig)

# 1. Количество патентов по годам
fig, ax = plt.subplots(figsize=(10, 5))
year_counts.plot(kind="bar", color="skyblue", ax=ax)
ax.set_title("Количество патентов по годам")
ax.set_xlabel("Год")
ax.set_ylabel("Количество патентов")
plt.xticks(rotation=45)
plt.tight_layout()
save_plot(fig, "patents_by_year")

# 2. Топ 10 авторов
fig, ax = plt.subplots(figsize=(8, 8))
top_authors = author_counts.head(10)
top_authors.plot(
    kind="pie",
    autopct="%1.1f%%",
    startangle=140,
    colors=plt.cm.tab20.colors,
    ax=ax
)
ax.set_ylabel("")
ax.set_title("Топ 10 авторов по количеству патентов")
plt.tight_layout()
save_plot(fig, "top_10_authors_pie")

# 3. Тепловая карта (Авторы × Годы)
top_authors = heatmap_data.sum(axis=1).nlargest(10).index
heatmap_top = heatmap_data.loc[top_authors]
fig, ax = plt.subplots(figsize=(14, 8))
sns.heatmap(heatmap_top, cmap="YlGnBu", linewidths=0.5, annot=True, fmt="d", ax=ax)
ax.set_title("Тепловая карта: Авторы × Годы")
ax.set_xlabel("Год")
ax.set_ylabel("Автор")
plt.tight_layout()
save_plot(fig, "heatmap_authors_years")

# 4. Количество патентов по областям
fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(
    x=area_counts.index,
    y=area_counts.values,
    hue=area_counts.index,
    palette="Set2",
    legend=False,
    ax=ax
)
ax.set_title("Количество патентов по областям")
ax.set_xlabel("Область научной деятельности")
ax.set_ylabel("Количество патентов")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
save_plot(fig, "patents_by_area")

# 5. Стековая гистограмма Годы × Области
fig, ax = plt.subplots(figsize=(14, 7))
year_area_counts.plot(kind="bar", stacked=True, colormap="tab20", ax=ax)
ax.set_title("Количество патентов по годам и областям")
ax.set_xlabel("Год")
ax.set_ylabel("Количество патентов")
plt.xticks(rotation=45)
plt.tight_layout()
save_plot(fig, "patents_year_area_stacked")

# 6. Тепловая карта Области × Годы
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(year_area_counts.T, cmap="YlGnBu", linewidths=0.5, annot=True, fmt="d", ax=ax)
ax.set_title("Тепловая карта: Области × Годы")
ax.set_xlabel("Год")
ax.set_ylabel("Область")
plt.tight_layout()
save_plot(fig, "heatmap_area_year")

# ===============================
# Статистика
# ===============================
print("Общее количество патентов:", len(df))
print("Количество уникальных авторов:", df_exploded["Авторы"].nunique())
print(f"Все графики сохранены в папку: {OUTPUT_DIR.resolve()}")
