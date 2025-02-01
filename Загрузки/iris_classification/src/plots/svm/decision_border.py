import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

def svm_plot_pca(kernel, x_train, y_train, x_test, y_test, classes, save_dir="plots/svm"):
    '''
    Визуализация границ принятия решениий SVM после PCA 
    '''

    os.makedirs(save_dir, exist_ok=True)

    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    # Применяем PCA
    pca = PCA(n_components=2)
    X_train_2d = pca.fit_transform(x_train)
    X_test_2d = pca.fit_transform(x_test)

    # Обучаем модель
    model = SVC(kernel=kernel)
    model.fit(X_train_2d, y_train_encoded)

    y_pred = model.predict(X_test_2d)

    plt.figure(figsize=(6, 4))

    # Отражаем тестовые данные
    for class_value in classes:
        plt.scatter(
            X_test_2d[y_test == class_value, 0],  # Первый главный компонент
            X_test_2d[y_test == class_value, 1],  # Второй главный компонент
            label=f'{class_value}',
            alpha=0.7
        )

    # Создание сетки для отображения границы принятия решения
    x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
    y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Прогноз классов для каждого элемента сетки
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Отображение границы принятия решения
    plt.contourf(xx, yy, Z, alpha=0.2, levels=[-1, 0, 1, 2], colors=['blue', 'orange', 'green'])
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"SVM Decision Boundary ({kernel} kernel)")
    plt.legend(loc="best")

    # Сохраняем и закрываем
    plt.savefig(os.path.join(save_dir, f"svm_{kernel}_decision_boundary.png"), bbox_inches="tight")
    plt.close()
