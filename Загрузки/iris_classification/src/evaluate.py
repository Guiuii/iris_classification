from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import os

def metrics(method, y_test, y_pred):
    """
    Выводит метрики классификации
    """
    # Загрузка конфига
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f'{method}. Метрики для классификации:\n')
    print(f'Accuracy:  {accuracy_score(y_test, y_pred):.3f}')
    print(f'Precision:  {precision_score(y_test, y_pred, average="macro"):.3}')
    print(f'Recall:  {recall_score(y_test, y_pred, average="macro"):.3}')
    print('\nОтчет о классификации:')
    print(classification_report(y_test, y_pred))


def plot_confusion_matrix(method, y_test, y_pred, classes, save_dir):
    """
    Визуализирует confusion matrix.

    Параметры:
        method (str): Название метода (например, "Линейная регрессия").
        y_test (array-like): Фактические значения целевой переменной.
        y_pred (array-like): Предсказанные значения целевой переменной.
        classes (array-like): Уникальные классы (например, виды Iris).
    """

    os.makedirs(save_dir, exist_ok=True)
    # Вычисление матрицы
    cm = confusion_matrix(y_test, y_pred)

    # Визуализация
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        cmap='flare',
        fmt='d',  # Целочисленный формат
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{method} Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{method}_confusion_matrix.png"), bbox_inches="tight")
    plt.close()

