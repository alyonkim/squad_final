# Программный проект "SQUAD"

### Что мы имеем:
  1) контекст – некий параграф с текстом
  2) вопрос к этому контексту, подразумевающий ответ в виде подстроки контекста

### Цель:
  реализовать нейросеть, которая ищет этот ответ
  
### Источники:
  Следующая [статья](https://arxiv.org/pdf/1704.00051.pdf)

### Требования:
  python3 + tensorflow, numpy, msgpack, random, spacy, os.path, wget.  
  Подразумевается счёт на GPU и предварительное скачивание библиотеки для spacy:
  ```
  pip3 -m spacy download en_core_web_sm
  ```

### Инструкция
  1) constants.py – константы, используемые во всём проекте (параметры данных и параметры обучения, пути к файлам)
  2) prepare.py – все функции, используемые в других скриптах.
  3) train.py - скрипт, запускающий обучение
  4) test.py - скрипт, считающий качество (F1-score) на случайном батче тест-данных
  5) demo.py - скрипт, выдающий ответ по введённым пользователем данным  
    
  Необходимо загрузить все скрипты и все файлы из папки biases и запустить необходимый скрипт.
  
### Результат
  Использовались преобработанные данные от [facebook](https://github.com/facebookresearch/DrQA).  
  Обучение проводилось на 10 эпохах с размером батча 64.  
  Наблюдается падение качества через после третьей эпохи.  
  <pre>Среднее качество на трёх эпохах: 0.38014158743418286</pre>  
  <pre>Лучшее качество на трёх эпохах: 0.5477097004138335</pre>
