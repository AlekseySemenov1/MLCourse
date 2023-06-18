# Ансамблевые методы
В данном упражнении вам предстоит решить уже знакомую задачу классификации изображений – отделить изображения кошек от изображений собак, используя ансамбль моделей на основе стекинга.

В [предложенном архиве](https://drive.google.com/drive/folders/1tE4VvTyGxs_pyDRCrlH3cXD2uRLrzjY2?usp=drive_link) находится выборка, включающая в себя изображения кошек и собак (по 500 изображений). Имя каждого изображения, для удобства, имеет следующий формат: cat/dog.номер_изображения.jpg в зависимости от того, какое животное присутствует на изображении. Данная выборка используется для обучения классификатора и его оценки.

[Следующая выборка](https://drive.google.com/drive/folders/1iiyT6sHJb-1OMmkTvr-OqVo6XSly6trj?usp=drive_link) предназначена для классификации новых объектов после построения классификатора.

Важно! Используйте версию библиотеки scikit-learn==0.23.0, также Вам потребуется библиотека opencv
```python3
!pip install --upgrade pip
!pip install imutils
!pip install opencv-python
!pip install --upgrade scikit-learn==0.23.0
```
Для работы с изображениями и получения их гистограмм — характеристик распределения интенсивности изображения можно воспользоваться следующей функцией и библиотекой cv2:
```python3
def extract_histogram(image, bins=(8, 8, 8)):
    hist = cv2.calcHist([image], [0, 1, 2], None, bins, [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()
```
В задачах используйте реализацию алгоритмов из библиотеки sklearn:
```python3
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
```
## Часть 1
Обучите базовые алгоритмы на исходном наборе данных:

* Классификатор с мягким зазором и параметрами: C = 1.75, random_state = 195, остальные параметры по умолчанию;
* Бэггинг деревьев принятия решений. Параметры дерева: criterion = 'entropy', min_samples_leaf = 10, max_leaf_nodes = 20, random_state = 195, остальные параметры по умолчанию. Параметры бэггинга: n_estimators = 12, random_state = 195;
* Случайный лес с параметрами: n_estimators = 12, criterion = 'entropy', min_samples_leaf = 10, max_leaf_nodes = 20, random_state = 195, остальные параметры по умолчанию.

Обучите метаалгоритм — логистическая регрессия: solver='lbfgs', random_state = 195, остальные параметры по умолчанию. Оцените его точность при cv = 2.

Обучите модель стэкинга. Используйте 2-fold (cv = 2) кросс-валидацию для оценки.

1) Введите долю правильной классификации (Accuracy).

Выполните предсказание для изображений, указанных ниже и определите вероятность отнесения изображений к классу 1 (cat).

2) Файл dog.1032.jpg
3) Файл cat.1048.jpg
4) Файл dog.1024.jpg
5) Файл cat.1016.jpg
