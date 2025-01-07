
QUEEN:
=================
Сложность: 5
-----------------
Игра "Ферзь" представляет собой стратегическую игру для двух игроков, где каждый игрок управляет ферзем на шахматной доске размером 8x8. Цель игры - достичь противоположного края доски первым. Игроки поочередно перемещают своего ферзя, при этом ферзь может перемещаться по горизонтали, вертикали или диагонали на любое количество клеток.

Правила игры:
1. Два игрока управляют ферзями на шахматной доске размером 8x8.
2. Каждый игрок начинает игру со своим ферзем на одном из противоположных краев доски.
3. Игроки по очереди перемещают своих ферзей.
4. Ферзь может перемещаться по горизонтали, вертикали или диагонали на любое количество клеток.
5. Цель игры - первым достичь противоположного края доски.
6. Игра завершается, когда один из ферзей достигает противоположного края.
-----------------
Алгоритм:
1. Установить начальные координаты ферзей для игрока 1 (X1, Y1) и игрока 2 (X2, Y2).
2. Вывести на экран шахматную доску, с указанием текущего положения ферзей.
3. Начать цикл "пока один из ферзей не достигнет противоположной стороны":
    3.1 Запросить у игрока 1 координаты для перемещения ферзя (NX, NY).
    3.2 Проверить, является ли ход игрока 1 допустимым (ферзь может ходить только по прямой).
    3.3 Если ход не допустим, то сообщить об этом и запросить новый ход.
    3.4 Обновить координаты ферзя игрока 1 (X1 = NX, Y1 = NY).
    3.5 Проверить, не достиг ли ферзь игрока 1 противоположной стороны.
    3.6 Если достиг, то объявить о победе игрока 1 и закончить игру.
    3.7 Вывести на экран шахматную доску, с указанием текущего положения ферзей.
    3.8 Запросить у игрока 2 координаты для перемещения ферзя (NX, NY).
    3.9 Проверить, является ли ход игрока 2 допустимым (ферзь может ходить только по прямой).
    3.10 Если ход не допустим, то сообщить об этом и запросить новый ход.
    3.11 Обновить координаты ферзя игрока 2 (X2 = NX, Y2 = NY).
    3.12 Проверить, не достиг ли ферзь игрока 2 противоположной стороны.
    3.13 Если достиг, то объявить о победе игрока 2 и закончить игру.
    3.14 Вывести на экран шахматную доску, с указанием текущего положения ферзей.
4. Конец игры.
-----------------
Блок-схема:
```mermaid
flowchart TD
    Start["Начало"] --> InitializeQueens["Инициализация:
    <code><b>
    player1X = 1, player1Y = 4<br>
    player2X = 8, player2Y = 4<br>
    </b></code>"]
    InitializeQueens --> DisplayBoard["Отобразить доску с ферзями"]
    DisplayBoard --> Player1TurnStart{"Начало хода игрока 1"}
    Player1TurnStart --> Player1Input["Запрос ввода хода игрока 1: <code><b>nextX, nextY</b></code>"]
    Player1Input --> Player1CheckMove{"Проверка допустимости хода <code><b>(nextX, nextY)</b></code> для игрока 1"}
    Player1CheckMove -- Недопустимый ход --> Player1InvalidMove["Вывод сообщения: Недопустимый ход"]
    Player1InvalidMove --> Player1TurnStart
    Player1CheckMove -- Допустимый ход --> Player1UpdatePosition["Обновление позиции ферзя игрока 1: <code><b>player1X = nextX, player1Y = nextY</b></code>"]
    Player1UpdatePosition --> Player1CheckWin{"Проверка: достиг ли ферзь игрока 1 конца доски?"}
    Player1CheckWin -- Да --> Player1Win["Вывод сообщения: Игрок 1 победил!"]
    Player1Win --> End["Конец"]
    Player1CheckWin -- Нет --> DisplayBoardAfterPlayer1["Отобразить доску с новой позицией ферзя игрока 1"]
    DisplayBoardAfterPlayer1 --> Player2TurnStart{"Начало хода игрока 2"}
    Player2TurnStart --> Player2Input["Запрос ввода хода игрока 2: <code><b>nextX, nextY</b></code>"]
    Player2Input --> Player2CheckMove{"Проверка допустимости хода <code><b>(nextX, nextY)</b></code> для игрока 2"}
     Player2CheckMove -- Недопустимый ход --> Player2InvalidMove["Вывод сообщения: Недопустимый ход"]
     Player2InvalidMove --> Player2TurnStart
    Player2CheckMove -- Допустимый ход --> Player2UpdatePosition["Обновление позиции ферзя игрока 2: <code><b>player2X = nextX, player2Y = nextY</b></code>"]
    Player2UpdatePosition --> Player2CheckWin{"Проверка: достиг ли ферзь игрока 2 конца доски?"}
     Player2CheckWin -- Да --> Player2Win["Вывод сообщения: Игрок 2 победил!"]
    Player2Win --> End
    Player2CheckWin -- Нет --> DisplayBoardAfterPlayer2["Отобразить доску с новой позицией ферзя игрока 2"]
    DisplayBoardAfterPlayer2 --> Player1TurnStart

```

Legenda:
    Start - Начало программы.
    InitializeQueens - Инициализация начальных позиций ферзей обоих игроков.
    DisplayBoard - Отображение шахматной доски с текущими позициями ферзей.
    Player1TurnStart - Начало хода игрока 1.
    Player1Input - Запрос у игрока 1 координат следующего хода.
    Player1CheckMove - Проверка допустимости хода игрока 1.
    Player1InvalidMove - Вывод сообщения о недопустимом ходе для игрока 1.
    Player1UpdatePosition - Обновление позиции ферзя игрока 1 на доске.
    Player1CheckWin - Проверка, достиг ли ферзь игрока 1 конца доски.
    Player1Win - Вывод сообщения о победе игрока 1.
    DisplayBoardAfterPlayer1 - Отображение доски после хода игрока 1.
    Player2TurnStart - Начало хода игрока 2.
    Player2Input - Запрос у игрока 2 координат следующего хода.
    Player2CheckMove - Проверка допустимости хода игрока 2.
    Player2InvalidMove - Вывод сообщения о недопустимом ходе для игрока 2.
     Player2UpdatePosition - Обновление позиции ферзя игрока 2 на доске.
    Player2CheckWin - Проверка, достиг ли ферзь игрока 2 конца доски.
    Player2Win - Вывод сообщения о победе игрока 2.
     DisplayBoardAfterPlayer2 - Отображение доски после хода игрока 2.
    End - Конец программы.
"""
import sys

# Инициализация начальных позиций ферзей
player1_x = 0
player1_y = 3
player2_x = 7
player2_y = 3


def print_board(player1_x, player1_y, player2_x, player2_y):
    """
    Выводит на экран шахматную доску с указанием текущих позиций ферзей.
    """
    print("   0  1  2  3  4  5  6  7")
    for row in range(8):
        row_str = str(row) + " "
        for col in range(8):
            if row == player1_y and col == player1_x:
                row_str += " 1 "
            elif row == player2_y and col == player2_x:
                row_str += " 2 "
            else:
                row_str += " . "
        print(row_str)


def is_valid_move(current_x, current_y, next_x, next_y):
    """
    Проверяет, является ли ход ферзя допустимым.

    Ход допустим, если ферзь двигается по горизонтали, вертикали или диагонали.
    """
    if next_x < 0 or next_x > 7 or next_y < 0 or next_y > 7:
        return False # Проверка выхода за границы доски
    
    if current_x == next_x: # Вертикальное перемещение
        return True
    elif current_y == next_y: # Горизонтальное перемещение
        return True
    elif abs(current_x - next_x) == abs(current_y - next_y): # Диагональное перемещение
         return True
    else:
        return False


def get_player_move(player_number, current_x, current_y):
    """
    Запрашивает у игрока ввод координат для перемещения ферзя.
    Проверяет допустимость введенных координат.
    Возвращает новые координаты.
    """
    while True:
        try:
            move_str = input(f"Игрок {player_number}, введите ход (x, y): ")
            next_x, next_y = map(int, move_str.split(','))
            if is_valid_move(current_x, current_y, next_x, next_y):
                return next_x, next_y
            else:
                print("Недопустимый ход. Попробуйте еще раз.")
        except ValueError:
            print("Неверный формат ввода. Введите два числа через запятую, например: 1,2")


# Основной игровой цикл
while True:
    print_board(player1_x, player1_y, player2_x, player2_y) # Выводим доску
    # Ход игрока 1
    print("Ход игрока 1:")
    next_player1_x, next_player1_y = get_player_move(1, player1_x, player1_y)
    player1_x, player1_y = next_player1_x, next_player1_y

    if player1_x == 7: # Проверка на победу
        print("Игрок 1 победил!")
        break # Завершаем цикл
    # Ход игрока 2
    print_board(player1_x, player1_y, player2_x, player2_y)
    print("Ход игрока 2:")
    next_player2_x, next_player2_y = get_player_move(2, player2_x, player2_y)
    player2_x, player2_y = next_player2_x, next_player2_y

    if player2_x == 0: # Проверка на победу
       print("Игрок 2 победил!")
       break # Завершаем цикл


"""
Объяснение кода:

1. **Инициализация переменных**:
   - `player1_x`, `player1_y`: Начальные координаты ферзя игрока 1.
   - `player2_x`, `player2_y`: Начальные координаты ферзя игрока 2.

2. **Функция `print_board`**:
   - Принимает текущие координаты ферзей обоих игроков.
   - Выводит на экран шахматную доску, используя '.' для пустых клеток, '1' для ферзя игрока 1 и '2' для ферзя игрока 2.

3. **Функция `is_valid_move`**:
   - Принимает текущие и следующие координаты для хода ферзя.
   - Проверяет, является ли ход допустимым:
     - Ферзь может двигаться по горизонтали, вертикали или диагонали.
     - Ферзь не может выйти за границы доски.

4. **Функция `get_player_move`**:
    - Запрашивает у игрока ввод координат следующего хода.
    - Использует цикл `while True`, чтобы гарантировать ввод правильных данных.
    - Проверяет формат ввода: ввод должен быть в формате `x,y` (два целых числа через запятую).
    - Вызывает функцию `is_valid_move` для проверки допустимости хода.
    - Возвращает новые координаты хода, когда они допустимы.

5. **Основной игровой цикл (`while True`)**:
   - Выводит шахматную доску на экран.
   - Запрашивает ход у игрока 1, используя функцию `get_player_move`, и обновляет позицию ферзя игрока 1.
   - Проверяет, не достиг ли ферзь игрока 1 конца доски (x=7). Если достиг, выводит сообщение о победе и завершает игру.
   - Выводит доску еще раз.
   - Запрашивает ход у игрока 2, используя функцию `get_player_move`, и обновляет позицию ферзя игрока 2.
   - Проверяет, не достиг ли ферзь игрока 2 конца доски (x=0). Если достиг, выводит сообщение о победе и завершает игру.
   - Цикл продолжается до тех пор, пока один из игроков не выиграет.

6.  **Запуск игры**:
    -  Игра запускается напрямую, без дополнительных условий.
"""
```