# Логистическая регрессия
В задачах используйте реализацию логистической регрессии из библиотеки sklearn:

from sklearn.linear_model import LogisticRegression

При обучении используйте следующие параметры: random_state = 2019, solver = 'lbfgs':

LogisticRegression(random_state = 2019, solver = 'lbfgs').fit(X, y)
## Часть 1
В [прилагаемом файле](/Data/LogisticRegressionTrain.csv) представлены данные, собранные путем голосования за самые лучшие (или, по крайней мере, самые популярные) конфеты Хэллоуина. Обучите модель логистической регрессии. В качестве предикторов выступают поля: chocolate, fruity, caramel, peanutyalmondy, nougat, crispedricewafer, hard, bar, pluribus, sugarpercent, pricepercent, отклик — Y.

В качестве тренировочного набора данных используйте данные из файла, за иключением следующих конфет: Dots, Fun Dip, Milky Way Midnight. Обучите модель.

Обучите модель и выполните предсказание для всех конфет из [прилагаемого файла](/Data/LogisticRegressionTest.csv) тестовых данных.
1) Введите вероятность отнесения конфеты Twix к классу 1.
2) Введите вероятность отнесения конфеты Tootsie Roll Juniors к классу 1.
Выполните оценку модели с помощью матрицы ошибок и рассчитайте следующие параметры при пороге отсечения (Treshhold) 0.5.
3) Введите значение Recall, или TPR для тестового набора данных.
4) Введите значение Precision для тестового набора данных.
5) Введите значение AUC для тестового набора данных.
