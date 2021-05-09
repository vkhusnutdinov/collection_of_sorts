from timeit import timeit
from random import randint
from functools import total_ordering
from bisect import bisect_left
from itertools import zip_longest
from heapq import merge
import operator
import random


# Реализация на python встроенной сортировки tim_sort
def tim_sort(array):
    min_run = 32
    n = len(array)
    for i in range(0, n, min_run):
        insertion_sort(array, i, min((i + min_run - 1), n - 1))
    size = min_run
    while size < n:
        for start in range(0, n, size * 2):
            midpoint = start + size - 1
            end = min((start + size * 2 - 1), (n-1))
            merged_array = merge_s(
                left=array[start:midpoint + 1],
                right=array[midpoint + 1:end + 1])
            array[start:start + len(merged_array)] = merged_array
        size *= 2
    return array


def insertion_sort(array, left=0, right=None):
    if right is None:
        right = len(array) - 1
    for i in range(left + 1, right + 1):
        key_item = array[i]
        j = i - 1
        while j >= left and array[j] > key_item:
            array[j + 1] = array[j]
            j -= 1
        array[j + 1] = key_item
    return array


# Сортировка пузырьком
def bubble_sort1(a):
    """
    Алгоритм состоит из повторяющихся проходов по сортируемому массиву. За каждый проход элементы последовательно
    сравниваются попарно и, если порядок в паре неверный, выполняется обмен элементов. Проходы по массиву повторяются
    N-1 раз или до тех пор, пока на очередном проходе не окажется, что обмены больше не нужны, что означает — массив
    отсортирован. При каждом проходе алгоритма по внутреннему циклу, очередной наибольший элемент массива ставится на
    своё место в конце массива рядом с предыдущим «наибольшим элементом», а наименьший элемент перемещается на одну
    позицию к началу массива («всплывает» до нужной позиции, как пузырёк в воде — отсюда и название алгоритма).

    Сложность:  O(n^2).
    """
    n = len(a)
    for bypass in range(1, n):
        for k in range(0, n-bypass):
            if a[k] > a[k+1]:
                a[k], a[k+1] = a[k+1], a[k]
    return a


def bubble_sort2(array):
    n = len(array)
    for i in range(n):
        for j in range(n - i - 1):
            already_sorted = True
            if array[j] > array[j + 1]:
                array[j], array[j + 1] = array[j + 1], array[j]
                already_sorted = False
        if already_sorted:
            break

    return array


# Сортировка перемешиванием
def cocktail_sort(a):
    """
    Алгоритм расширяет пузырьковую сортировку, работая в двух направлениях. Хотя он улучшает пузырьковую сортировку
    за счет более быстрого перемещения элементов в начало списка , он обеспечивает лишь незначительное улучшение
    производительности.

    Сложность:  O(n^2).
    """
    left = 0
    right = len(a) - 1

    while left <= right:
        for i in range(left, right):
            if a[i] > a[i+1]:
                a[i], a[i+1] = a[i+1], a[i]
        right -= 1
        for i in range(right, left, -1):
            if a[i-1] > a[i]:
                a[i], a[i-1] = a[i-1], a[i]
        left += 1
    return a


# Сортировка чет-нечет
def odd_even(data):
    """
    Является модификацией пузырьковой сортировки.

    Заводится флаг, определяющий отсортирован ли массив. В начале итерации ставится в состояние «истина»,
    далее каждый нечётный элемент сверяется с последующим и если они стоят в неправильном порядке (предыдущий больше
    следующего), то они меняются местами, и флаг ставится в состояние «ложь». То же самое делается с чётными
    элементами. Алгоритм не прекращает работу, пока флаг не останется в состоянии «истина».

    Сложность: O(n^2).
    Устойчивость: да.
    """
    n = len(data)
    is_sorted = 0
    while is_sorted == 0:
        is_sorted = 1
        for i in range(1, n - 1, 2):
            if data[i] > data[i + 1]:
                data[i], data[i + 1] = data[i + 1], data[i]
                is_sorted = 0

        for i in range(0, n - 1, 2):
            if data[i] > data[i + 1]:
                data[i], data[i + 1] = data[i + 1], data[i]
                is_sorted = 0
    return data


# Сортировка расческой
def comb_sort1(data):
    """
    Основная идея алгоритма в том, чтобы первоначально брать достаточно большое расстояние между сравниваемыми
    элементами и по мере упорядочивания массива сужать это расстояние вплоть до минимального. Таким образом,
    мы как бы причёсываем массив, постепенно разглаживая на всё более аккуратные пряди. Сначала расстояние между
    элементами максимально, то есть равно размеру массива минус один. Затем, пройдя массив с этим шагом, необходимо
    поделить шаг на фактор уменьшения и пройти по списку вновь. Так продолжается до тех пор, пока разность индексов
    не достигнет единицы. В этом случае сравниваются соседние элементы как и в сортировке пузырьком,
    но такая итерация одна.

    Сложность: O(n^2).
    Устойчивость: нет.
    """
    alen = len(data)
    gap = (alen * 10 // 13) if alen > 1 else 0
    while gap:
        if 8 < gap < 11:    # variant "comb-11"
            gap = 11
        swapped = False
        for i in range(alen - gap):
            if data[i + gap] < data[i]:
                data[i], data[i + gap] = data[i + gap], data[i]
                swapped = True
        gap = (gap * 10 // 13) or swapped


# Сортировка расческой, другой вариант
def comb_sort2(data):
    gap = len(data)
    swaps = True
    while gap > 1 or swaps:
        gap = max(1, int(gap / 1.3))  # minimum gap is 1
        swaps = False
        for i in range(len(data) - gap):
            j = i + gap
            if data[i] > data[j]:
                data[i], data[j] = data[j], data[i]
                swaps = True
    return data


# Гномья сортировка
def gnome_sort(a):
    """
    Алгоритм находит первое место, где два соседних элемента стоят в неправильном порядке и меняет их местами. Он
    пользуется тем фактом, что обмен может породить новую пару, стоящую в неправильном порядке, только до или после
    переставленных элементов. Он не допускает, что элементы после текущей позиции отсортированы, таким образом,
    нужно только проверить позицию до переставленных элементов.

    Сложность: O(n^2); рекурсивная версия требует дополнительно O(n^2) памяти.
    Устойчивость: да.
    """
    for i in range(len(a)):
        while i > 0 and a[i-1] > a[i]:
            a[i], a[i-1] = a[i-1], a[i]
            i -= 1
    return a


# Быстрая сортировка
def quick_sort1(data):
    """
    Является существенно улучшенным вариантом алгоритма сортировки с помощью прямого обмена (таких как
    «Пузырьковая сортировка» и «Шейкерная сортировка»). Принципиальное отличие состоит в том, что в первую очередь
    производятся перестановки на наибольшем возможном расстоянии и после каждого прохода элементы делятся на две
    независимые группы.

    Сложность: в варианте с минимальными затратами памяти — сложность алгоритма: O(n log n) — среднее время,
    O(n^2) — худший случай.
    Устойчивость: нет.
    """
    less = []
    pivot_list = []
    more = []
    if len(data) <= 1:
        return data
    else:
        pivot = data[0]
        for i in data:
            if i < pivot:
                less.append(i)
            elif i > pivot:
                more.append(i)
            else:
                pivot_list.append(i)
        less = quick_sort1(less)
        more = quick_sort1(more)
        return less + pivot_list + more


# Быстрая сортировка, решение через рандомное нахождние опорного элемента


def quick_sort2(nums):
    if len(nums) <= 1:
        return nums
    else:
        q = random.choice(nums)
    l_nums = [n for n in nums if n < q]
    e_nums = [q] * nums.count(q)
    b_nums = [n for n in nums if n > q]
    return quick_sort2(l_nums) + e_nums + quick_sort2(b_nums)


def quick_sort3(array):
    if len(array) < 2:
        return array
    low, same, high = [], [], []
    pivot = array[randint(0, len(array) - 1)]
    for item in array:
        if item < pivot:
            low.append(item)
        elif item == pivot:
            same.append(item)
        elif item > pivot:
            high.append(item)
    return quick_sort3(low) + same + quick_sort3(high)


# Сортировка вставками
def insertion_sort1(a):
    """
    Алгоритм сортировки, в котором элементы входной последовательности просматриваются по одному, и каждый новый
    поступивший элемент размещается в подходящее место среди ранее упорядоченных элементов.

    Сложность:  O(n^2).
    Устойчивость: да.
    """
    for top in range(1, len(a)):
        k = top
        while k > 0 and a[k-1] > a[k]:
            a[k], a[k-1] = a[k-1], a[k]
            k -= 1
    return a


# Сортировка вставками, улучшенный вариант
def insertion_sort2(data):
    for i in range(1, len(data)):
        item_to_insert = data[i]
        j = i - 1
        while j >= 0 and data[j] > item_to_insert:
            data[j+1] = data[j]
            j -= 1
        data[j+1] = item_to_insert
    return data


# Сортировка вставками с бинарным поиском
def insertion_binary(data):
    """
    Место для вставки производится с помощью бинарного поиска.

    Сложность:  O(n^2 / 2).
    Устойчивость: да.
    """
    for i in range(1, len(data) - 1):
        key = data[i]
        lo, hi = 0, i - 1
        while lo < hi:
            mid = lo + (hi - lo) // 2
            if key < data[mid]:
                hi = mid
            else:
                lo = mid + 1
        for j in range(i, lo + 1, -1):
            data[j] = data[j - 1]
        data[lo] = key
    return data


# Сортировка Шелла
def shell_sort(data):
    """
    Улучшение сортировки вставками.

    В отличие от простых вставок сортировка Шелла не пытается слева от элемента сразу формировать строго
    отсортированную часть массива. Она создаёт слева от элемента почти отсортированную часть массива и делает это
    достаточно быстро.

    Сложность:  меняется в зависимости от выбора последовательности длин промежутков; при определённом выборе,
    возможно обеспечить сложность O(n^(4/3)) или O(n log^2 n).
    Устойчивость: нет.
    """
    inc = len(data) // 2
    while inc:
        for i, el in enumerate(data):
            while i >= inc and data[i - inc] > el:
                data[i] = data[i - inc]
                i -= inc
            data[i] = el
        inc = 1 if inc == 2 else int(inc * 5.0 / 11)
    return data


# Пасьянсная сортировка
@total_ordering
class Pile(list):
    def __lt__(self, other): return self[-1] < other[-1]
    def __eq__(self, other): return self[-1] == other[-1]


def patience_sort(n):
    """
    Название алгоритма происходит от упрощенного варианта карточной игры.

    Сложность:  O(n log n).
    Устойчивость: _.
    """
    piles = []
    for x in n:
        new_pile = Pile([x])
        i = bisect_left(piles, new_pile)
        if i != len(piles):
            piles[i].append(x)
        else:
            piles.append(new_pile)

    n[:] = merge(*[reversed(pile) for pile in piles])
    return n


# Сортировка выбором
def selection_sort1(a):
    """
    Алгоритм делит входной список на две части: отсортированный подсписок элементов, который создается слева
    направо в начале (слева) списка, и подсписок оставшихся несортированных элементов, которые занимают остальную
    часть списка. Первоначально отсортированный подсписок пуст, а несортированный подсписок представляет собой весь
    входной список. Алгоритм продолжается путем нахождения наименьшего (или наибольшего, в зависимости от порядка
    сортировки) элемента в несортированном подсписке, обмена (перестановки) его с крайним левым несортированным
    элементом (помещая его в отсортированном порядке) и перемещения границ подсписка на один элемент вправо.

    Сложность:  O(n^2).
    Устойчивость: нет.
    """
    n = len(a)
    for pos in range(0, n-1):
        for k in range(pos+1, n):
            if a[k] < a[pos]:
                a[k], a[pos] = a[pos], a[k]
    return a


# Сортировка выбором, улучшенный вариант
def selection_sort2(data):
    for i in range(len(data)):
        lowest = i
        for j in range(i + 1, len(data)):
            if data[j] < data[lowest]:
                lowest = j
        data[i], data[lowest] = data[lowest], data[i]
    return data


# Сортировка выбором, лучший вариант
def selection_sort3(data):
    for i, e in enumerate(data):
        mn = min(range(i, len(data)), key=data.__getitem__)
        data[i], data[mn] = data[mn], e
    return data


# Сортировка с одновременным выбором минимального и максимального значения
def double_selection(arr):
    n = len(arr)
    i = 0
    j = n - 1
    while i < j:
        min_a = arr[i]
        max_a = arr[i]
        min_i = i
        max_i = i
        for k in range(i, j + 1, 1):
            if arr[k] > max_a:
                max_a = arr[k]
                max_i = k
            elif arr[k] < min_a:
                min_a = arr[k]
                min_i = k
        temp = arr[i]
        arr[i] = arr[min_i]
        arr[min_i] = temp
        if arr[min_i] == max_a:
            temp = arr[j]
            arr[j] = arr[min_i]
            arr[min_i] = temp
        else:
            temp = arr[j]
            arr[j] = arr[max_i]
            arr[max_i] = temp
        i += 1
        j -= 1
    return arr


# Бинго-сортировка
def bingo_sort(data):
    """
    Модификация простой сортировки выбором, позволяющая быстрее сортировать массивы, состоящие из неуникальных
    элементов.

    Сложность:  O(n^2 / 2).
    Устойчивость: нет.
    """
    mx = len(data) - 1
    next_value = data[mx]
    for i in range(mx - 1, -1, -1):
        if data[i] > next_value:
            next_value = data[i]
    while mx and data[mx] == next_value:
        mx -= 1
    while mx:
        value = next_value
        next_value = data[mx]
        for i in range(mx - 1, -1, -1):
            if data[i] == value:
                data[i], data[mx] = data[mx], data[i]
                mx -= 1
            elif data[i] > next_value:
                next_value = data[i]
        while mx and data[mx] == next_value:
            mx -= 1
    return data


def cycle_sort(data):
    """
    Цикличная сортировка интересна (и ценна с практической точки зрения) тем, что изменения среди элементов массива
    происходят тогда и только тогда, когда элемент ставится на своё конечное место. Это может пригодиться,
    если перезапись в массиве — слишком дорогое удовольствие.

    Сложность:  O(n^2 / 2).
    Устойчивость: нет.
    """
    for cycleStart in range(0, len(data) - 1):
        value = data[cycleStart]
        pos = cycleStart
        for i in range(cycleStart + 1, len(data)):
            if data[i] < value:
                pos += 1
        if pos == cycleStart:
            continue
        while value == data[pos]:
            pos += 1
        data[pos], value = value, data[pos]
        while pos != cycleStart:
            pos = cycleStart
            for i in range(cycleStart + 1, len(data)):
                if data[i] < value:
                    pos += 1
            while value == data[pos]:
                pos += 1
            data[pos], value = value, data[pos]
    return data


def pancake_sort(data):
    """
    Ищем максимальный элемент. Переворачиваем цепочку элементов от левого края до максимума - в результате максимум
    оказывается на левом крае. Затем переворачиваем весь неотсортированный подмассив, в результате чего максимум
    попадает на своё место. Эти действия повторяем с оставшейся неотсортированной частью массива.

    Сложность:  O(n)*.
    Устойчивость: да.
    """
    if len(data) > 1:
        for size in range(len(data), 1, -1):
            max_index = max(range(size), key=data.__getitem__)
            if max_index + 1 != size:
                if max_index != 0:
                    data[:max_index + 1] = reversed(data[:max_index + 1])
                data[:size] = reversed(data[:size])
    return data


# Пирамидальная сортировка
def heap_sort(nums):
    """
    Сначала преобразуем список в Max Heap — бинарное дерево, где самый большой элемент является вершиной дерева.
    Затем помещаем этот элемент в конец списка. После перестраиваем Max Heap и снова помещаем новый наибольший
    элемент уже перед последним элементом в списке. Этот процесс построения кучи повторяется, пока все вершины дерева
    не будут удалены.

    Сложность:  O (n log n).
    Устойчивость: нет.
    """
    n = len(nums)
    for i in range(n, -1, -1):
        heapify(nums, n, i)
    for i in range(n - 1, 0, -1):
        nums[i], nums[0] = nums[0], nums[i]
        heapify(nums, i, 0)


def heapify(data, heap_size, r_idx):
    largest = r_idx
    left_child = (2 * r_idx) + 1
    right_child = (2 * r_idx) + 2
    if left_child < heap_size and data[left_child] > data[largest]:
        largest = left_child
    if right_child < heap_size and data[right_child] > data[largest]:
        largest = right_child
    if largest != r_idx:
        data[r_idx], data[largest] = data[largest], data[r_idx]
        heapify(data, heap_size, largest)


# Сортировка слиянием
def merge_sort1(data, compare=operator.lt):
    """
    Сначала задача разбивается на несколько подзадач меньшего размера. Затем эти задачи решаются с помощью
    рекурсивного вызова или непосредственно, если их размер достаточно мал. Наконец, их решения комбинируются,
    и получается решение исходной задачи.

    Сложность:  O(n log n). Требуется O(n) дополнительной памяти.
    Устойчивость: да.
    """
    if len(data) < 2:
        return data[:]
    else:
        middle = int(len(data) / 2)
        left = merge_sort1(data[:middle], compare)
        right = merge_sort1(data[middle:], compare)
        result = []
        i, j = 0, 0
        while i < len(left) and j < len(right):
            if compare(left[i], right[j]):
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        while i < len(left):
            result.append(left[i])
            i += 1
        while j < len(right):
            result.append(right[j])
            j += 1
        return result


# Сортировка слиянием, усложненный вариант
def merge_sort2(array):
    if len(array) < 2:
        return array
    midpoint = len(array) // 2
    return merge_s(
        left=merge_sort2(array[:midpoint]),
        right=merge_sort2(array[midpoint:]))


def merge_s(left, right):
    if len(left) == 0:
        return right
    if len(right) == 0:
        return left

    result = []
    index_left = index_right = 0

    while len(result) < len(left) + len(right):
        if left[index_left] <= right[index_right]:
            result.append(left[index_left])
            index_left += 1
        else:
            result.append(right[index_right])
            index_right += 1
        if index_right == len(right):
            result += left[index_left:]
            break
        if index_left == len(left):
            result += right[index_right:]
            break
    return result


# Сортировка слиянием, решение через кучу
def merge_sort3(m):
    if len(m) <= 1:
        return m

    middle = len(m) // 2
    left = m[:middle]
    right = m[middle:]

    left = merge_sort3(left)
    right = merge_sort3(right)
    return list(merge(left, right))


# Сортировка прямым слиянием Боуза-Нельсона
def bose_nelson(data):
    m = 1
    while m < len(data):
        j = 0
        while j + m < len(data):
            bose_nelson_merge(j, m, m, data)
            j = j + m + m
        m = m + m
    return data


def bose_nelson_merge(j, r, m, data):
    if j + r < len(data):
        if m == 1:
            if data[j] > data[j + r]:
                data[j], data[j + r] = data[j + r], data[j]
        else:
            m = m // 2
            bose_nelson_merge(j, r, m, data)
            if j + r + m < len(data):
                bose_nelson_merge(j + m, r, m, data)
            bose_nelson_merge(j + m, r - m, m, data)
    return data


# Нитевидная сортировка
def strand_sort(a):
    """
    Алгоритм сначала перемещает первый элемент списка в подсписок. Затем сравнивается последний элемент в подсписке с
    каждым последующим элементом в исходном списке. Как только в исходном списке есть элемент, который больше,
    чем последний элемент в подсписке, этот элемент удаляется из исходного списка и добавляется в подсписок. Этот
    процесс продолжается до тех пор, пока последний элемент в подсписке не будет сравнен с остальными элементами в
    исходном списке. Подсписок затем объединяется в новый список. Процесс повторяется объедения все подсписки,
    пока все элементы не будут отсортированы.

    Сложность: O(n^2).
    Устойчивость: да.
    """
    out = strand(a)
    while len(a):
        out = merge_list(out, strand(a))
    return out


def merge_list(a, b):
    out = []
    while len(a) and len(b):
        if a[0] < b[0]:
            out.append(a.pop(0))
        else:
            out.append(b.pop(0))
    out += a
    out += b
    return out


def strand(a):
    i, s = 0, [a.pop(0)]
    while i < len(a):
        if a[i] > s[-1]:
            s.append(a.pop(i))
        else:
            i += 1
    return s


# Блочная сортировка
def bucket_sort(input_list):
    """
    Алгоритм сортировки, в котором сортируемые элементы распределяются между конечным числом отдельных блоков (
    карманов, корзин) так, чтобы все элементы в каждом следующем по порядку блоке были всегда больше (или меньше),
    чем в предыдущем. Каждый блок затем сортируется отдельно, либо рекурсивно тем же методом, либо другим. Затем
    элементы помещаются обратно в массив.

    Сложность: требуется O(k) дополнительной памяти и знание о природе сортируемых данных, выходящее за рамки функций
    «переставить» и «сравнить». Сложность алгоритма: O(n).
    """
    max_value = max(input_list)
    size = max_value / len(input_list)
    buckets_list = []
    for x in range(len(input_list)):
        buckets_list.append([])
    for i in range(len(input_list)):
        j = int(input_list[i] / size)
        if j != len(input_list):
            buckets_list[j].append(input_list[i])
        else:
            buckets_list[len(input_list) - 1].append(input_list[i])
    for z in range(len(input_list)):
        insertion_sort2(buckets_list[z])
    final_output = []
    for x in range(len(input_list)):
        final_output = final_output + buckets_list[x]
    return final_output


# Сортировка подсчетом
def counting_sort(a):
    """
    Алгоритм сортировки, в котором используется диапазон чисел сортируемого массива (списка) для подсчёта
    совпадающих элементов. Применение сортировки подсчётом целесообразно лишь тогда, когда сортируемые числа имеют (
    или их можно отобразить в) диапазон возможных значений, который достаточно мал по сравнению с сортируемым
    множеством, например, миллион натуральных чисел меньших 1000.

    Сложность: O(n+k). Требуется O(n+k) дополнительной памяти.
    """
    max_el = max(a)
    cnt = [0] * (max_el + 1)

    for i in range(len(a)):
        cnt[a[i]] += 1

    pos = 0
    for num in range(len(cnt)):
        for i in range(cnt[num]):
            a[pos] = num
            pos += 1
    return a


# Бисерная сортировка так же известная как gravity sort
def bead_sort(data):
    return list(map(sum, zip_longest(*[[1] * e for e in data], fillvalue=0)))


# Поразрядная сортировка
def radix_sort(a):
    """
    Сравнение производится поразрядно: сначала сравниваются значения одного крайнего разряда, и элементы
    группируются по результатам этого сравнения, затем сравниваются значения следующего разряда, соседнего,
    и элементы либо упорядочиваются по результатам сравнения значений этого разряда внутри образованных на предыдущем
    проходе групп, либо переупорядочиваются в целом, но сохраняя относительный порядок, достигнутый при предыдущей
    сортировке. Затем аналогично делается для следующего разряда, и так до конца.

    Сложность: O(nk); требуется O(k) дополнительной памяти.
    """
    length = len(str(max(a)))
    rang = 10
    for i in range(length):
        b = [[] for _ in range(rang)]
        for x in a:
            figure = x // 10 ** i % 10
            b[figure].append(x)
        a = []
        for k in range(rang):
            a = a + b[k]
    return a


if __name__ == '__main__':
    data_array = ['sorted', 'tim_sort', 'bubble_sort1', 'bubble_sort2', 'cocktail_sort', 'odd_even', 'comb_sort1',
                  'comb_sort2', 'gnome_sort', 'quick_sort1', 'quick_sort2', 'quick_sort3', 'insertion_sort1',
                  'insertion_sort2', 'insertion_binary', 'shell_sort', 'heap_sort', 'patience_sort', 'selection_sort1',
                  'selection_sort2', 'selection_sort3', 'double_selection', 'bingo_sort', 'cycle_sort',
                  'pancake_sort', 'merge_sort1', 'merge_sort2', 'merge_sort3', 'bose_nelson', 'strand_sort',
                  'bucket_sort', 'counting_sort', 'bead_sort', 'radix_sort']
    unsorted = [randint(1, 100) for _ in range(1000)]
    result = 0
    for ns in data_array:
        if ns == data_array[0]:
            time_sorted = timeit(f'{ns}(j)', setup='j = unsorted.copy()', number=1, globals=globals())
        else:
            result = timeit(f'{ns}(j)', setup='j = unsorted.copy()', number=1, globals=globals())
        print('%.19f  ..%-17s  %.3f %%' % (result or time_sorted, ns, time_sorted / result * 100 if result else 100))


