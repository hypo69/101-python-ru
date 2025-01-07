"""
ACE:
=================
Сложность: 7
-----------------
Игра "ACE" - это игра, в которой два игрока по очереди вытягивают карты из колоды и пытаются набрать больше очков. Туз считается за 1 очко, а карты с номерами от 2 до 10 считаются по номиналу, а также валет, дама и король считаются за 10. Игрок, набравший больше очков, побеждает. Игра продолжается до тех пор, пока не будет разыграно определённое количество раундов.

Правила игры:
1. Играют два игрока.
2. Игроки по очереди вытягивают карты из колоды.
3. Каждая карта имеет определенное количество очков: туз - 1, карты от 2 до 10 - по номиналу, валет, дама и король - 10.
4. Каждый игрок старается набрать как можно больше очков за раунд.
5. В конце раунда сравниваются очки игроков.
6. Игра состоит из определенного количества раундов.
7. Победителем игры объявляется игрок, набравший больше очков за все раунды.
-----------------
Алгоритм:
1. Инициализация очков игроков 1 и 2 нулями.
2. Запросить количество раундов.
3. Начать цикл по количеству раундов:
    3.1. Игрок 1 вытягивает карту.
    3.2. Вывести карту игрока 1 и количество очков за карту.
    3.3. Добавить очки за карту к общему количеству очков игрока 1.
    3.4. Игрок 2 вытягивает карту.
    3.5. Вывести карту игрока 2 и количество очков за карту.
    3.6. Добавить очки за карту к общему количеству очков игрока 2.
    3.7. Если очки игрока 1 больше очков игрока 2, вывести сообщение "PLAYER 1 WINS THE ROUND".
    3.8. Если очки игрока 2 больше очков игрока 1, вывести сообщение "PLAYER 2 WINS THE ROUND".
    3.9. Если очки игрока 1 равны очкам игрока 2, вывести сообщение "TIE GAME THIS ROUND".
4. Вывести общее количество очков игрока 1.
5. Вывести общее количество очков игрока 2.
6. Если общее количество очков игрока 1 больше общего количества очков игрока 2, вывести сообщение "PLAYER 1 WINS THE GAME".
7. Если общее количество очков игрока 2 больше общего количества очков игрока 1, вывести сообщение "PLAYER 2 WINS THE GAME".
8. Если общее количество очков игрока 1 равно общему количеству очков игрока 2, вывести сообщение "TIE GAME".
9. Конец игры.
-----------------
Блок-схема:
```mermaid
flowchart TD
    Start["Начало"] --> InitializeScores["<p align='left'>Инициализация переменных:<br><code><b>player1Score = 0</b></code><br><code><b>player2Score = 0</b></code></p>"]
    InitializeScores --> InputRounds["Ввод количества раундов: <code><b>numberOfRounds</b></code>"]
    InputRounds --> RoundLoopStart{"Начало цикла по раундам"}
    RoundLoopStart -- Да --> Player1DrawsCard["Игрок 1 тянет карту: <code><b>card1, card1Value</b></code>"]
    Player1DrawsCard --> OutputPlayer1Card["Вывод карты и очков игрока 1: <code><b>card1, card1Value</b></code>"]
    OutputPlayer1Card --> UpdatePlayer1Score["<code><b>player1Score = player1Score + card1Value</b></code>"]
    UpdatePlayer1Score --> Player2DrawsCard["Игрок 2 тянет карту: <code><b>card2, card2Value</b></code>"]
    Player2DrawsCard --> OutputPlayer2Card["Вывод карты и очков игрока 2: <code><b>card2, card2Value</b></code>"]
    OutputPlayer2Card --> UpdatePlayer2Score["<code><b>player2Score = player2Score + card2Value</b></code>"]
    UpdatePlayer2Score --> CompareScores{"Сравнение очков за раунд: <code><b>card1Value > card2Value?</b></code>"}
    CompareScores -- Да --> OutputPlayer1RoundWin["Вывод: <b>PLAYER 1 WINS THE ROUND</b>"]
    CompareScores -- Нет --> CompareScores2{"Сравнение очков за раунд: <code><b>card2Value > card1Value?</b></code>"}
    CompareScores2 -- Да --> OutputPlayer2RoundWin["Вывод: <b>PLAYER 2 WINS THE ROUND</b>"]
    CompareScores2 -- Нет --> OutputTieRound["Вывод: <b>TIE GAME THIS ROUND</b>"]
    OutputPlayer1RoundWin --> RoundLoopEnd
    OutputPlayer2RoundWin --> RoundLoopEnd
    OutputTieRound --> RoundLoopEnd
     RoundLoopEnd --> RoundLoopStart {"Начало цикла по раундам"}

    RoundLoopStart -- Нет --> OutputTotalPlayer1Score["Вывод общего количества очков игрока 1: <code><b>player1Score</b></code>"]
    OutputTotalPlayer1Score --> OutputTotalPlayer2Score["Вывод общего количества очков игрока 2: <code><b>player2Score</b></code>"]
    OutputTotalPlayer2Score --> CompareTotalScores{"Сравнение общих очков: <code><b>player1Score > player2Score?</b></code>"}
    CompareTotalScores -- Да --> OutputPlayer1GameWin["Вывод: <b>PLAYER 1 WINS THE GAME</b>"]
    CompareTotalScores -- Нет --> CompareTotalScores2{"Сравнение общих очков: <code><b>player2Score > player1Score?</b></code>"}
     CompareTotalScores2 -- Да --> OutputPlayer2GameWin["Вывод: <b>PLAYER 2 WINS THE GAME</b>"]
    CompareTotalScores2 -- Нет --> OutputTieGame["Вывод: <b>TIE GAME</b>"]
    OutputPlayer1GameWin --> End["Конец"]
    OutputPlayer2GameWin --> End
    OutputTieGame --> End
```
**Legenda**
    Start - Начало программы.
    InitializeScores - Инициализация переменных очков игроков player1Score и player2Score нулями.
    InputRounds - Запрос у пользователя количества раундов numberOfRounds для игры.
    RoundLoopStart - Начало цикла для каждого раунда игры. Цикл выполняется numberOfRounds раз.
    Player1DrawsCard - Игрок 1 вытягивает карту card1 и определяется ее значение card1Value.
    OutputPlayer1Card - Вывод на экран информации о карте игрока 1 card1 и ее значении card1Value.
    UpdatePlayer1Score - Обновление общего счета игрока 1, добавляя к player1Score значение card1Value.
    Player2DrawsCard - Игрок 2 вытягивает карту card2 и определяется ее значение card2Value.
    OutputPlayer2Card - Вывод на экран информации о карте игрока 2 card2 и ее значении card2Value.
    UpdatePlayer2Score - Обновление общего счета игрока 2, добавляя к player2Score значение card2Value.
    CompareScores - Сравнение значений карт card1Value и card2Value для определения победителя раунда.
    OutputPlayer1RoundWin - Вывод сообщения о победе игрока 1 в раунде.
    CompareScores2 - Сравнение значений карт card2Value и card1Value для определения победителя раунда.
    OutputPlayer2RoundWin - Вывод сообщения о победе игрока 2 в раунде.
    OutputTieRound - Вывод сообщения о ничьей в раунде.
    RoundLoopEnd - Конец цикла по раундам.
    OutputTotalPlayer1Score - Вывод на экран общего количества очков игрока 1 player1Score.
    OutputTotalPlayer2Score - Вывод на экран общего количества очков игрока 2 player2Score.
    CompareTotalScores - Сравнение общего счета игроков player1Score и player2Score для определения победителя игры.
    OutputPlayer1GameWin - Вывод сообщения о победе игрока 1 в игре.
     CompareTotalScores2 - Сравнение общего счета игроков player2Score и player1Score для определения победителя игры.
    OutputPlayer2GameWin - Вывод сообщения о победе игрока 2 в игре.
    OutputTieGame - Вывод сообщения о ничьей в игре.
    End - Конец программы.
"""
import random

def calculate_card_value(card):
    """
    Вычисляет значение карты. Туз - 1, валет, дама, король - 10, остальные по номиналу.
    """
    if card in ['J', 'Q', 'K']:
        return 10
    elif card == 'A':
        return 1
    else:
        try:
            return int(card)
        except ValueError:
            return 0

def draw_card(deck):
    """
    Вытягивает случайную карту из колоды.
    """
    card = random.choice(deck)
    return card, calculate_card_value(card)

def play_ace_game():
    """
    Основная функция игры ACE.
    """
    player1Score = 0  # Инициализация счета игрока 1
    player2Score = 0  # Инициализация счета игрока 2
    
    # Создание колоды карт
    deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4
    
    try:
        numberOfRounds = int(input("Сколько раундов вы хотите сыграть? ")) # Запрос количества раундов у пользователя
        if numberOfRounds <= 0:
            print("Количество раундов должно быть положительным числом.")
            return
    except ValueError:
        print("Пожалуйста, введите целое число для количества раундов.")
        return

    for roundNumber in range(1, numberOfRounds + 1):
      print(f"\nРаунд {roundNumber}:")

      # Игрок 1 вытягивает карту
      card1, card1Value = draw_card(deck)
      print(f"Игрок 1 вытащил {card1} ({card1Value} очков)")
      player1Score += card1Value  # Обновление счета игрока 1

      # Игрок 2 вытягивает карту
      card2, card2Value = draw_card(deck)
      print(f"Игрок 2 вытащил {card2} ({card2Value} очков)")
      player2Score += card2Value  # Обновление счета игрока 2

      # Сравнение очков за раунд
      if card1Value > card2Value:
          print("Игрок 1 выиграл этот раунд")
      elif card2Value > card1Value:
          print("Игрок 2 выиграл этот раунд")
      else:
          print("Ничья в этом раунде")
    
    # Вывод общего счета
    print(f"\nОбщий счет:")
    print(f"Игрок 1: {player1Score} очков")
    print(f"Игрок 2: {player2Score} очков")

    # Определение победителя игры
    if player1Score > player2Score:
        print("Игрок 1 выиграл игру!")
    elif player2Score > player1Score:
        print("Игрок 2 выиграл игру!")
    else:
        print("Ничья в игре!")


if __name__ == "__main__":
    play_ace_game()
"""
Объяснение кода:
1.  **Импорт модуля `random`**:
    - `import random`: Импортирует модуль `random`, который используется для генерации случайных карт.
2.  **Функция `calculate_card_value(card)`**:
    -   Принимает карту в качестве аргумента.
    -   Возвращает числовое значение карты.
        -   Для карт 'J', 'Q', 'K' возвращает 10.
        -   Для карты 'A' возвращает 1.
        -   Для остальных карт возвращает их номинал (преобразуя строку в целое число).
        -   Обрабатывает ошибку ValueError, если карта не распознана, возвращает 0.
3.  **Функция `draw_card(deck)`**:
    -   Принимает колоду карт в качестве аргумента.
    -   Выбирает случайную карту из колоды с помощью `random.choice()`.
    -   Возвращает карту и ее числовое значение.
4.  **Функция `play_ace_game()`**:
    -   `player1Score = 0` и `player2Score = 0`: Инициализирует счетчики очков для игроков.
    -   `deck = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A'] * 4`: Создает стандартную колоду из 52 карт.
    -   Запрашивает у пользователя количество раундов и проверяет ввод на корректность (чтобы было целое положительное число).
    -   **Основной цикл `for roundNumber in range(1, numberOfRounds + 1):`**:
        -   Цикл выполняется для каждого раунда игры.
        -   **Карта игрока 1**:
            -   `card1, card1Value = draw_card(deck)`: Игрок 1 вытягивает карту, и определяется её значение.
            -   Выводится информация о карте и ее значении.
            -   `player1Score += card1Value`: Значение карты добавляется к общему счету игрока 1.
        -   **Карта игрока 2**:
            -   `card2, card2Value = draw_card(deck)`: Игрок 2 вытягивает карту, и определяется её значение.
            -    Выводится информация о карте и ее значении.
            -   `player2Score += card2Value`: Значение карты добавляется к общему счету игрока 2.
        -   **Сравнение очков**:
            -   Сравниваются значения карт игроков в текущем раунде.
            -   Выводится сообщение о победе одного из игроков или о ничьей в раунде.
    -   **Вывод общего счета**:
        -   Выводит общий счет каждого игрока.
    -   **Определение победителя игры**:
        -   Сравниваются общие очки игроков.
        -   Выводится сообщение о победе одного из игроков или о ничьей в игре.
5.  **Запуск игры**:
    -   `if __name__ == "__main__":`: Этот блок гарантирует, что функция `play_ace_game()` будет запущена только если файл исполняется напрямую, а не импортируется как модуль.
    -   `play_ace_game()`: Вызывает функцию для начала игры.
"""
