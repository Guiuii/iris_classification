from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_logistic(x_train, y_train, model_save_path):
    """Обучение и сохранение логистической регрессии"""
    os.makedirs(model_save_path, exist_ok=True)
    
    model = LogisticRegression(multi_class='ovr')
    model.fit(x_train, y_train)
    
    # Сохраняем модель
    joblib.dump(model, os.path.join(model_save_path, "logistic_model.pkl"))
    return model

def predict_logistic(model, x_test):
    """Прогнозирование с помощью обученной модели"""
    return model.predict(x_test)
