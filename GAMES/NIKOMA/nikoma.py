"""
ROCKET:
=================
Сложность: 7
-----------------
Игра "Ракета" - это игра, в которой игрок управляет ракетой, пытаясь достичь заданной высоты, расходуя топливо. Цель игры - достичь высоты 100, прежде чем закончится топливо. Игрок может выбирать, сколько топлива потратить на каждом шаге. 
В зависимости от выбора игрок поднимается на случайную высоту.

Правила игры:
1. Игрок начинает с 100 единицами топлива.
2. Игрок должен достичь высоты 100.
3. На каждом шаге игрок вводит количество топлива, которое он хочет потратить на подъем.
4. Подъем ракеты зависит от потраченного топлива и является случайным числом от 0 до 2*топливо.
5. Игра заканчивается, когда игрок достигает высоты 100 или когда заканчивается топливо.
-----------------
Алгоритм:
1. Установить начальную высоту ракеты в 0.
2. Установить начальное количество топлива в 100.
3. Начать цикл "пока высота ракеты не равна 100 и топливо больше 0":
   3.1 Вывести текущую высоту ракеты и остаток топлива.
   3.2 Запросить у игрока количество топлива для сжигания.
   3.3 Если введенное количество топлива больше имеющегося, вывести сообщение "Вы не имеете столько топлива".
   3.4 Иначе:
     3.4.1 Уменьшить количество топлива на введенное значение.
     3.4.2 Вычислить подъем ракеты как случайное целое число от 0 до 2 * введенное значение топлива.
     3.4.3 Увеличить текущую высоту ракеты на вычисленный подъем.
4. Если высота ракеты равна 100, вывести сообщение "Поздравляю! Вы достигли высоты 100!".
5. Иначе, если топливо равно 0, вывести сообщение "Вы исчерпали топливо. Вы не достигли высоты 100!".
6. Конец игры.
-----------------
Блок-схема:
```mermaid
flowchart TD
    Start["Начало"] --> InitializeVariables["<p align='left'>Инициализация переменных:
    <code><b>
    rocketHeight = 0
    fuel = 100
    </b></code></p>"]
    InitializeVariables --> LoopStart{"Начало цикла: 
    пока <code><b>rocketHeight < 100</b></code> И <code><b>fuel > 0</b></code>"}
    LoopStart -- Да --> OutputStatus["Вывод: 
    <code><b>
    Высота: {rocketHeight}
    Топливо: {fuel}
    </b></code>"]
    OutputStatus --> InputFuel["Запрос ввода топлива: <code><b>fuelToBurn</b></code>"]
    InputFuel --> CheckFuel["Проверка: <code><b>fuelToBurn > fuel?</b></code>"]
    CheckFuel -- Да --> OutputError["Вывод: <b>Вы не имеете столько топлива!</b>"]
    OutputError --> LoopStart
    CheckFuel -- Нет --> BurnFuel["<code><b>fuel = fuel - fuelToBurn</b></code>"]
    BurnFuel --> CalculateRise["Вычисление подъема:
    <code><b>
    rise = random(0, 2 * fuelToBurn)
    </b></code>"]
    CalculateRise --> IncreaseHeight["<code><b>rocketHeight = rocketHeight + rise</b></code>"]
    IncreaseHeight --> CheckWin{"Проверка: <code><b>rocketHeight >= 100</b></code>"}
    CheckWin -- Да --> OutputWin["Вывод: <b>Поздравляю! Вы достигли высоты 100!</b>"]
    OutputWin --> End["Конец"]
    CheckWin -- Нет --> LoopStart
     LoopStart -- Нет --> CheckFuelEmpty{"Проверка: <code><b>fuel == 0</b></code>"}
    CheckFuelEmpty -- Да --> OutputLose["Вывод: <b>Вы исчерпали топливо. Вы не достигли высоты 100!</b>"]
    OutputLose --> End
     CheckFuelEmpty -- Нет --> End

```
**Legenda:**
    Start - Начало игры.
    InitializeVariables - Инициализация переменных: `rocketHeight` (высота ракеты) устанавливается в 0, а `fuel` (количество топлива) устанавливается в 100.
    LoopStart - Начало цикла, который продолжается, пока высота ракеты меньше 100 и количество топлива больше 0.
    OutputStatus - Вывод текущей высоты ракеты и остатка топлива.
    InputFuel - Запрос у пользователя количества топлива для сжигания.
    CheckFuel - Проверка, больше ли запрашиваемое количество топлива, чем доступное.
    OutputError - Вывод сообщения об ошибке, если запрашиваемое количество топлива больше доступного.
    BurnFuel - Уменьшение количества топлива на введенное значение.
    CalculateRise - Вычисление подъема ракеты как случайного числа от 0 до 2 * введенное количество топлива.
    IncreaseHeight - Увеличение высоты ракеты на вычисленный подъем.
    CheckWin - Проверка, достигла ли ракета высоты 100.
    OutputWin - Вывод сообщения о победе, если ракета достигла высоты 100.
    End - Конец игры.
    CheckFuelEmpty - Проверка, закончилось ли топливо.
    OutputLose - Вывод сообщения о проигрыше, если топливо закончилось.
"""
import random

# Инициализация высоты ракеты и количества топлива
rocketHeight = 0
fuel = 100

# Основной игровой цикл
while rocketHeight < 100 and fuel > 0:
    print(f"Высота: {rocketHeight}, Топливо: {fuel}")
    try:
        fuelToBurn = int(input("Сколько топлива сжечь?: "))
    except ValueError:
        print("Пожалуйста, введите целое число.")
        continue
    if fuelToBurn > fuel:
        print("У вас нет столько топлива!")
    else:
        fuel -= fuelToBurn
        rise = random.randint(0, 2 * fuelToBurn)
        rocketHeight += rise

# Проверка условий окончания игры и вывод сообщения
if rocketHeight >= 100:
    print("Поздравляю! Вы достигли высоты 100!")
else:
    print("Вы исчерпали топливо. Вы не достигли высоты 100!")

"""
Объяснение кода:
1.  **Импорт модуля `random`**:
    -   `import random`: Импортирует модуль `random`, который используется для генерации случайного числа.
2.  **Инициализация переменных**:
    -   `rocketHeight = 0`: Инициализирует переменную `rocketHeight` для хранения текущей высоты ракеты.
    -   `fuel = 100`: Инициализирует переменную `fuel` для хранения текущего количества топлива.
3.  **Основной игровой цикл `while rocketHeight < 100 and fuel > 0:`**:
    -   Цикл продолжается, пока ракета не достигнет высоты 100 или не закончится топливо.
    -   `print(f"Высота: {rocketHeight}, Топливо: {fuel}")`: Выводит текущую высоту и количество топлива.
    -   **Ввод данных**:
        - `try...except ValueError`: Блок try-except обрабатывает возможные ошибки ввода. Если пользователь введет не целое число, то будет выведено сообщение об ошибке.
        -   `fuelToBurn = int(input("Сколько топлива сжечь?: "))`: Запрашивает у пользователя количество топлива для сжигания.
    -   **Проверка наличия топлива**:
        -   `if fuelToBurn > fuel:`: Проверяет, достаточно ли топлива для сжигания.
        -   `print("У вас нет столько топлива!")`: Выводит сообщение об ошибке, если топлива недостаточно.
    -   **Сжигание топлива и расчет подъема**:
        -   `else:`: Если топлива достаточно, то выполняется следующий блок.
        -   `fuel -= fuelToBurn`: Уменьшает количество топлива на потраченное количество.
        -   `rise = random.randint(0, 2 * fuelToBurn)`: Вычисляет случайный подъем в диапазоне от 0 до 2 * `fuelToBurn`.
        -   `rocketHeight += rise`: Увеличивает высоту ракеты на величину подъема.
4.  **Проверка условий окончания игры и вывод сообщения**:
    -   `if rocketHeight >= 100:`: Проверяет, достигла ли ракета высоты 100.
    -   `print("Поздравляю! Вы достигли высоты 100!")`: Выводит сообщение о победе, если ракета достигла высоты 100.
    -   `else:`: Если ракета не достигла высоты 100.
    -   `print("Вы исчерпали топливо. Вы не достигли высоты 100!")`: Выводит сообщение о проигрыше, если закончилось топливо.
"""
