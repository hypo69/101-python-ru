"""
<LIT QZ>:
=================
Сложность: 4
-----------------
Игра "LIT QZ" представляет собой викторину, где компьютер задает вопросы, а игрок должен на них ответить. 
В оригинальной версии игры вопросы и ответы закодированы в виде данных, но мы можем сделать викторину более интерактивной и расширяемой, 
позволяя легко добавлять новые вопросы и ответы. Игра проверяет знания игрока, позволяя пройти через серию вопросов. 

Правила игры:
1. Компьютер выводит вопрос.
2. Игрок вводит свой ответ.
3. Компьютер проверяет ответ и сообщает, был ли ответ верным.
4. Игра продолжается до тех пор, пока не закончатся вопросы.
5. В конце игры выводится сообщение о ее завершении.
-----------------
Алгоритм:
1. Инициализировать список вопросов и ответов.
2. Установить счетчик вопросов в 0.
3. Начать цикл "пока не кончатся вопросы":
   3.1. Вывести текущий вопрос.
   3.2. Запросить у игрока ввод ответа.
   3.3. Сравнить введенный ответ с правильным ответом.
   3.4. Если ответ верный, вывести сообщение "RIGHT".
   3.5. Если ответ неверный, вывести сообщение "WRONG".
   3.6. Увеличить счетчик вопросов на 1.
4. Вывести сообщение "THAT'S ALL FOLKS!"
5. Конец игры.
-----------------
"""

# Определение списка вопросов и ответов
questions = [
    ("What is the capital of France?", "Paris"),
    ("What is 2 + 2?", "4"),
    ("What is the largest planet in our solar system?", "Jupiter"),
    ("What is the chemical symbol for water?", "H2O"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci")
]

# Инициализация индекса текущего вопроса
questionIndex = 0

# Основной цикл игры
while questionIndex < len(questions):
    # Выводим текущий вопрос
    currentQuestion, correctAnswer = questions[questionIndex]
    print(currentQuestion)
    
    # Запрашиваем ввод ответа пользователя
    userAnswer = input("Your answer: ")
    
    # Проверяем правильность ответа
    if userAnswer.lower() == correctAnswer.lower():
        print("RIGHT")
    else:
        print("WRONG")
    
    # Увеличиваем индекс вопроса
    questionIndex += 1

# Выводим сообщение об окончании игры
print("THAT'S ALL FOLKS!")

"""
Объяснение кода:
1. **Инициализация вопросов и ответов**:
   - `questions = [...]`: Создается список `questions`, который содержит кортежи. Каждый кортеж представляет собой пару: вопрос (строка) и правильный ответ (строка).
   - `questionIndex = 0`: Инициализируется переменная `questionIndex` для отслеживания текущего вопроса, с которого начинается викторина.
2.  **Основной цикл `while questionIndex < len(questions):`**:
    -   Цикл продолжается, пока `questionIndex` меньше длины списка `questions`, то есть пока не будут заданы все вопросы.
    -   `currentQuestion, correctAnswer = questions[questionIndex]`: Извлекает текущий вопрос и правильный ответ из списка вопросов по текущему индексу.
    -   `print(currentQuestion)`: Выводит текущий вопрос на экран.
    -   **Ввод ответа**:
        - `userAnswer = input("Your answer: ")`: Запрашивает ввод ответа от пользователя и сохраняет его в переменной `userAnswer`.
    -   **Проверка ответа**:
        -   `if userAnswer.lower() == correctAnswer.lower():`: Сравнивает ответ пользователя (приведенный к нижнему регистру) с правильным ответом (также приведенным к нижнему регистру) для регистронезависимого сравнения.
        -   `print("RIGHT")`: Выводит сообщение "RIGHT", если ответ правильный.
        -   `else:`: Если ответ неправильный.
        -   `print("WRONG")`: Выводит сообщение "WRONG", если ответ неверный.
    -   `questionIndex += 1`: Увеличивает `questionIndex` на 1, чтобы перейти к следующему вопросу.
3.  **Вывод сообщения об окончании игры**:
    -  `print("THAT'S ALL FOLKS!")`: Выводит сообщение об окончании игры после того, как цикл закончится (т.е. все вопросы будут заданы).

"""
