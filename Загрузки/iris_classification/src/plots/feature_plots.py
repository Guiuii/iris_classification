import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_feature_plots(
    df: pd.DataFrame,
    save_dir: str = "plots",
    filename: str = "iris_features_plots.png"
) -> None:
    "Генерация парных графиков характеристик и сохранение в папку"
    os.makedirs(save_dir, exist_ok=True)

    features = df.drop(['Species'], axis=1).columns

    plt.figure(figsize=(12, 8))
    k = 1
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            plt.subplot(2, 3, k)
            sns.scatterplot(x=features[i],
                            y=features[j],
                            data=df,
                            hue="Species"
            )
            k += 1

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), bbox_inches="tight")
    plt.close()

