import pandas as pd
import os
from sklearn.model_selection import train_test_split

def load_and_clean_data(data_path: str) -> pd.DataFrame:
    """Загрузка данных и базовая очистка"""
    df = pd.read_csv(data_path)
    df = df.drop(columns=['Id'], axis=1)  # Удаление ненужного столбца
    return df

def split_features_target(df: pd.DataFrame) -> tuple:
    """Разделение на признаки и целевую переменную"""
    X = df.drop(columns=['Species'], axis=1)
    y = df['Species']
    return X, y

def preprocess_data(config: dict) -> tuple:
    """Основная функция предобработки"""
    
    # Для запуска main.py из любой дтректории
    script_path = os.path.dirname(__file__)
    data_path = script_path.split('/')[:-1]
    data_path = "/".join(data_path)
    dataset_path = os.path.join(data_path, config["data"]["path"])

    # Загрузка и очистка данных
    df = load_and_clean_data(dataset_path)
    
    # Разделение на признаки и таргет
    X, y = split_features_target(df)
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y,
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state'],
    )
    
    return X_train, X_test, y_train, y_test
