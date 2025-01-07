
<BOXING>:
=================
Сложность: 4
-----------------
Игра "Бокс" представляет собой текстовую имитацию боксерского поединка между двумя игроками, где каждый игрок по очереди наносит удары, выбирая их силу (слабый, средний, сильный), и компьютер случайным образом определяет, попал ли удар. Игра продолжается до тех пор, пока один из боксеров не будет нокаутирован (его счетчик здоровья не достигнет 0).
Правила игры:
1.  Играют два игрока по очереди.
2.  У каждого игрока есть счетчик здоровья, изначально равный 10.
3.  Игрок может выбрать один из трех типов удара: слабый (1), средний (2) или сильный (3).
4.  Сила удара влияет на шанс попадания: слабый удар имеет наибольшую вероятность попадания, сильный - наименьшую.
5.  Если удар попадает, то здоровье противника уменьшается на 1.
6.  Игра заканчивается, когда здоровье одного из игроков достигает 0.
7.  Игрок, у которого здоровье осталось больше 0, объявляется победителем.
-----------------
Алгоритм:
1. Инициализировать здоровье обоих игроков (игрок 1 и игрок 2) значением 10.
2. Начать игровой цикл, который продолжается, пока здоровье обоих игроков больше 0:
    2.1 Вывести текущее здоровье обоих игроков.
    2.2 Запросить ввод от текущего игрока (игрок 1 или 2).
        - Если текущий игрок - игрок 1, запросить ввод силы удара (1-слабый, 2-средний, 3-сильный).
        - Если текущий игрок - игрок 2, запросить ввод силы удара (1-слабый, 2-средний, 3-сильный).
    2.3 Сгенерировать случайное число от 1 до 100.
    2.4 Если:
        - игрок выбрал слабый удар (1) и случайное число меньше или равно 80, то удар достиг цели, уменьшить здоровье противника на 1.
        - игрок выбрал средний удар (2) и случайное число меньше или равно 60, то удар достиг цели, уменьшить здоровье противника на 1.
        - игрок выбрал сильный удар (3) и случайное число меньше или равно 40, то удар достиг цели, уменьшить здоровье противника на 1.
        - иначе, удар не достиг цели.
    2.5 Вывести сообщение, достиг ли удар цели.
    2.6 Переключить ход на следующего игрока.
3. После завершения цикла определить победителя (у кого здоровье больше 0).
4. Вывести сообщение о победителе.
-----------------
Блок-схема:
```mermaid
flowchart TD
    Start["Начало"] --> InitializeHealth["<p align='left'>Инициализация здоровья игроков:
    <code><b>
    player1Health = 10
    player2Health = 10
    currentPlayer = 1
    </b></code></p>"]
    InitializeHealth --> LoopStart{"Начало цикла: пока здоровье обоих игроков > 0"}
    LoopStart -- Да --> DisplayHealth["Вывод текущего здоровья игроков"]
    DisplayHealth --> InputAttack["Ввод силы удара игроком <code><b>currentPlayer</b></code>"]
    InputAttack --> GenerateRandomNumber["Генерация случайного числа <code><b>randomNum</b></code> от 1 до 100"]
    GenerateRandomNumber --> CheckHit["<p align='left'>Проверка на попадание в зависимости от силы удара:
    <code><b>
    if (attackType == 1 and randomNum <= 80) or
        (attackType == 2 and randomNum <= 60) or
        (attackType == 3 and randomNum <= 40)
    </b></code></p>"]
    CheckHit -- Да --> ReduceOpponentHealth["Уменьшение здоровья противника на 1"]
    ReduceOpponentHealth --> OutputHit["Вывод сообщения о попадании"]
    OutputHit --> SwitchPlayer["Переключение на следующего игрока"]
    SwitchPlayer --> LoopStart
    CheckHit -- Нет --> OutputMiss["Вывод сообщения о промахе"]
    OutputMiss --> SwitchPlayer
    LoopStart -- Нет --> DetermineWinner["Определение победителя"]
    DetermineWinner --> OutputWinner["Вывод сообщения о победителе"]
    OutputWinner --> End["Конец"]
```

Legenda:
    Start - Начало программы.
    InitializeHealth - Инициализация переменных: player1Health и player2Health (здоровье игроков) устанавливаются в 10, currentPlayer (текущий игрок) устанавливается в 1.
    LoopStart - Начало цикла, который продолжается, пока здоровье обоих игроков больше 0.
    DisplayHealth - Вывод текущего здоровья обоих игроков.
    InputAttack - Запрос у текущего игрока ввода силы удара (1, 2 или 3) и сохранение его в переменной attackType.
    GenerateRandomNumber - Генерация случайного числа в диапазоне от 1 до 100 и сохранение его в переменной randomNum.
    CheckHit - Проверка, попал ли удар, на основе случайного числа и выбранной силы удара.
    ReduceOpponentHealth - Уменьшение здоровья противника на 1, если удар попал.
    OutputHit - Вывод сообщения о попадании удара.
    SwitchPlayer - Переключение хода на следующего игрока.
    OutputMiss - Вывод сообщения о промахе.
    DetermineWinner - Определение победителя после окончания цикла.
    OutputWinner - Вывод сообщения о победителе.
    End - Конец программы.
```
```python
import random

# Инициализация здоровья игроков
player1Health = 10
player2Health = 10
# Начинаем с первого игрока
currentPlayer = 1


# Основной игровой цикл
while player1Health > 0 and player2Health > 0:
    # Выводим текущее здоровье игроков
    print(f"Здоровье игрока 1: {player1Health}, Здоровье игрока 2: {player2Health}")
    
    # Запрашиваем ввод от текущего игрока
    while True:
        try:
            attackType = int(input(f"Игрок {currentPlayer}, выберите силу удара (1-слабый, 2-средний, 3-сильный): "))
            if attackType in [1, 2, 3]:
                break
            else:
                print("Некорректный ввод. Пожалуйста, выберите 1, 2 или 3.")
        except ValueError:
            print("Некорректный ввод. Пожалуйста, введите число.")
    
    # Генерируем случайное число
    randomNum = random.randint(1, 100)
    
    # Определяем, попал ли удар и уменьшаем здоровье противника, если удар попал
    hit = False
    if attackType == 1 and randomNum <= 80:
        hit = True
    elif attackType == 2 and randomNum <= 60:
        hit = True
    elif attackType == 3 and randomNum <= 40:
        hit = True
    
    if hit:
        print("Удар достиг цели!")
        if currentPlayer == 1:
            player2Health -= 1
        else:
            player1Health -= 1
    else:
        print("Удар не достиг цели.")

    # Переключаем ход на следующего игрока
    currentPlayer = 3 - currentPlayer  # Переключает с 1 на 2 и с 2 на 1

# Определяем победителя
if player1Health <= 0:
    print("Игрок 2 победил!")
else:
    print("Игрок 1 победил!")
```
```
Объяснение кода:
1.  **Импорт модуля `random`**:
   -  `import random`: Импортирует модуль `random`, который используется для генерации случайных чисел.
2.  **Инициализация переменных**:
    -   `player1Health = 10`: Инициализирует здоровье игрока 1 со значением 10.
    -   `player2Health = 10`: Инициализирует здоровье игрока 2 со значением 10.
    -   `currentPlayer = 1`: Устанавливает текущего игрока на игрока 1.
3. **Основной игровой цикл `while player1Health > 0 and player2Health > 0:`**:
   -  Цикл продолжается, пока здоровье обоих игроков больше 0.
   -  **Вывод текущего здоровья**:
        - `print(f"Здоровье игрока 1: {player1Health}, Здоровье игрока 2: {player2Health}")`: Выводит текущее здоровье обоих игроков.
   - **Запрос ввода от текущего игрока**:
      -  `while True:`:  Бесконечный цикл для запроса ввода, пока не будет введен корректный вариант.
      -  `try...except ValueError:`: Обрабатывает возможные ошибки ввода (если пользователь введет не число).
      -  `attackType = int(input(f"Игрок {currentPlayer}, выберите силу удара (1-слабый, 2-средний, 3-сильный): "))`: Запрашивает у текущего игрока выбор силы удара.
      -  `if attackType in [1, 2, 3]:`: Проверяет, корректен ли ввод (1, 2 или 3).
      - `break`: Выходит из цикла запроса ввода при корректном вводе.
      - `else:`: Выводит сообщение о некорректном вводе, если введено число не 1, 2 или 3.
   - **Генерация случайного числа**:
      -  `randomNum = random.randint(1, 100)`: Генерирует случайное число от 1 до 100.
   - **Проверка попадания и уменьшение здоровья**:
      - `hit = False`: Инициализирует переменную hit (попадание) со значением False.
      - `if attackType == 1 and randomNum <= 80:`: Проверяет, если выбран слабый удар и случайное число меньше или равно 80, удар достигает цели.
      - `elif attackType == 2 and randomNum <= 60:`: Проверяет, если выбран средний удар и случайное число меньше или равно 60, удар достигает цели.
      - `elif attackType == 3 and randomNum <= 40:`: Проверяет, если выбран сильный удар и случайное число меньше или равно 40, удар достигает цели.
      - `if hit:`: Проверяет, был ли удар успешен.
      - `print("Удар достиг цели!")`: Выводит сообщение об успехе.
      -  Уменьшает здоровье противника в зависимости от текущего игрока.
      -  `else:`: Если удар не достиг цели.
      -  `print("Удар не достиг цели.")`: Выводит сообщение о неудаче.
   - **Переключение хода**:
      -   `currentPlayer = 3 - currentPlayer`: Переключает текущего игрока (если был 1, то становится 2, и наоборот).
4.  **Определение победителя**:
    -  `if player1Health <= 0:`: Проверяет, если здоровье игрока 1 стало 0 или меньше.
    - `print("Игрок 2 победил!")`: Выводит сообщение о победе игрока 2.
    -   `else:`: Если здоровье игрока 1 больше 0.
    -  `print("Игрок 1 победил!")`: Выводит сообщение о победе игрока 1.
```