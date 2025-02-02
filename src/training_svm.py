from sklearn.svm import SVC
import joblib
import os

def train_svm(x_train, y_train, model_save_path, kernel='linear'):
    """Обучение SVM с разными ядрами"""
    os.makedirs(model_save_path, exist_ok=True)
    
    params = {
        'linear': {'kernel': 'linear'},
        'rbf': {'kernel': 'rbf'},
        'poly': {'kernel': 'poly', 'degree': 3}
    }
    
    model = SVC(
        kernel=params[kernel]['kernel'],
        degree=params[kernel].get('degree', 3),
        decision_function_shape='ovr',
        probability=True
    )
    model.fit(x_train, y_train)
    
    # Сохранение модели
    joblib.dump(model, os.path.join(model_save_path, f"svm_{kernel}_model.pkl"))
    return model

def predict_svm(model, x_test):
    """Прогнозирование с помощью SVM"""
    return model.predict(x_test)
