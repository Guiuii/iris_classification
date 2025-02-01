import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.special import softmax
import joblib
import os

def train_linear(x_train, y_train, model_save_path="models/linear_regression"):
    """Обучение one-vs-all линейных регрессий"""
    os.makedirs(model_save_path, exist_ok=True)
    
    models = []
    classes = np.unique(y_train)
    
    # Обучаем по одной модели на каждый класс
    for idx, class_name in enumerate(classes):
        y_binary = (y_train == class_name).astype(int)
        model = LinearRegression()
        model.fit(x_train, y_binary)
        models.append(model)
        
        # Сохраняем каждую модель
        joblib.dump(model, os.path.join(model_save_path, f"{class_name}_model.pkl"))
    
    return models

def predict_linear(models, x_test):
    """Прогнозирование с ансамблем моделей"""
    predictions = np.zeros((x_test.shape[0], len(models)))
    
    for i, model in enumerate(models):
        predictions[:, i] = model.predict(x_test)
    
    # Применяем softmax для вероятностей
    probabilities = softmax(predictions, axis=1)
    classes = np.array(['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'])
    return classes[np.argmax(probabilities, axis=1)]
