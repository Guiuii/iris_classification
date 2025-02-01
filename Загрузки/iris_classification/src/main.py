from load_preprocessing import load_and_clean_data, split_features_target, preprocess_data
from training_linear import train_linear, predict_linear
from training_logistic import train_logistic, predict_logistic
from training_svm import train_svm, predict_svm
from evaluate import metrics, plot_confusion_matrix
from plots.feature_plots import generate_feature_plots
from plots.svm.decision_border import svm_plot_pca
import yaml
import os
import numpy as np
import pandas as pd

def main():
    #Загрузка конфига
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Загрузка и предобработка данных
    script_path = os.path.dirname(__file__)
    data_path = script_path.split('/')[:-1]
    data_path = "/".join(data_path)
    dataset_path = os.path.join(data_path, config["data"]["path"])
    df = load_and_clean_data(dataset_path)
    features, labels = split_features_target(df)
    x_train, x_test, y_train, y_test = preprocess_data(config)
    iris_classes = np.unique(y_train)

    # Генерация графиков, отражающих расположение элементов в пространстве признаков
    if config['preprocessing']['generate_plots']:
        generate_feature_plots(
        df=df,
        save_dir="plots",
        filename="iris_features_plot.png"
    )

    # Обучение и оценка моделей
    models = {
        'linear_regression': (train_linear, predict_linear),
        'logistic_regression': (train_logistic, predict_logistic),
        'svm': (train_svm, predict_svm)
    }

    for model_name, (train_func, predict_func) in models.items():
        if config['models'][model_name]['enable']:

            model_save_path = config['models'][model_name]['model_dir']

            # Обучение
            model = train_func(
                x_train,
                y_train,
                model_save_path=model_save_path
            )

            # Прогнозирование
            y_pred = predict_func(model, x_test)

            # Для SVM: дополнительная визуализация границ
            if model_name == 'svm':
                for kernel in config['models']['svm']['kernels']:
                    svm_plot_pca(
                    kernel=kernel,
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                    y_test=y_test,
                    classes=iris_classes,
                    save_dir=config['models']['svm']['plots_dir']
                )


            # Метрики
            metrics(model_name, y_test, y_pred)

            # Матрица ошибок
            plot_confusion_matrix(
              method=model_name,
              y_test=y_test,
              y_pred=y_pred,
              classes=labels.unique(),
              save_dir=f"plots/{model_name}"
    )

if __name__ == "__main__":
    main()
