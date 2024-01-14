import re
import time
import docx
import matplotlib.pyplot as plt
# Безпосередній пошук
def naive_search(text, words):
    found_words = {word: re.findall(word, text, re.IGNORECASE) for word in words}
    return found_words

# Класи для Ахо-Корасік
class TrieNode:
    def __init__(self):
        self.children = {}
        self.fail = None
        self.is_end_of_word = False
        self.output = []

class AhoCorasick:
    def __init__(self):
        self.root = TrieNode()

    def add_word(self, word):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_word = True
        node.output.append(word)

    def build_failure_pointers(self):
        queue = [self.root]
        while queue:
            current_node = queue.pop(0)
            for char, child_node in current_node.children.items():
                queue.append(child_node)
                fail_node = current_node.fail
                while fail_node and char not in fail_node.children:
                    fail_node = fail_node.fail
                child_node.fail = fail_node.children[char] if fail_node else self.root
                if child_node.fail:
                    child_node.output.extend(child_node.fail.output)

    def search(self, text):
        node = self.root
        results = {}
        for i in range(len(text)):
            char = text[i].lower()
            while node and char not in node.children:
                node = node.fail
            if not node:
                node = self.root
                continue
            node = node.children[char]
            if node.output:
                for word in node.output:
                    if word not in results:
                        results[word] = []
                    results[word].append(i - len(word) + 1)
        return results


def damerau_levenshtein_distance_with_path(s1, s2):
    d = {}
    lenstr1 = len(s1)
    lenstr2 = len(s2)
    path = {}

    for i in range(-1, lenstr1 + 1):
        d[(i, -1)] = i + 1
    for j in range(-1, lenstr2 + 1):
        d[(-1, j)] = j + 1

    for i in range(lenstr1):
        for j in range(lenstr2):
            if s1[i] == s2[j]:
                cost = 0
            else:
                cost = 1
            d[(i, j)] = min(
                d[(i - 1, j)] + 1,
                d[(i, j - 1)] + 1,
                d[(i - 1, j - 1)] + cost
            )
            path[(i, j)] = "substitution" if cost == 1 else "no operation"

            if i and j and s1[i] == s2[j - 1] and s1[i - 1] == s2[j]:
                if d[(i, j)] > d[i - 2, j - 2] + cost:
                    d[(i, j)] = d[i - 2, j - 2] + cost
                    path[(i, j)] = "transposition"

            if d[(i, j)] == d[(i - 1, j)] + 1:
                path[(i, j)] = "deletion"
            elif d[(i, j)] == d[(i, j - 1)] + 1:
                path[(i, j)] = "insertion"


    operations = []
    i, j = lenstr1 - 1, lenstr2 - 1
    while i >= 0 and j >= 0:
        operation = path[(i, j)]
        if operation == "deletion":
            operations.append(f"Delete '{s1[i]}' at position {i}")
            i -= 1
        elif operation == "insertion":
            operations.append(f"Insert '{s2[j]}' at position {i + 1}")
            j -= 1
        elif operation == "substitution":
            operations.append(f"Substitute '{s1[i]}' with '{s2[j]}' at position {i}")
            i -= 1
            j -= 1
        elif operation == "transposition":
            operations.append(f"Transpose '{s1[i - 1]}' and '{s1[i]}'")
            i -= 2
            j -= 2
        elif operation == "no operation":
            i -= 1
            j -= 1

    return d[lenstr1 - 1, lenstr2 - 1], operations[::-1]

def read_text_from_docx(filename):
    doc = docx.Document(filename)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def plot_search_results(naive_counts, aho_counts):
    labels = list(search_words)
    naive_values = [naive_counts.get(word, 0) for word in labels]
    aho_values = [aho_counts.get(word, 0) for word in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots()
    rects1 = ax.bar(x, naive_values, width, label='Безпосередній пошук')
    rects2 = ax.bar([p + width for p in x], aho_values, width, label='Ахо-Корасік')

    ax.set_ylabel('Кількість знайдених слів')
    ax.set_title('Порівняння методів пошуку')
    ax.set_xticks([p + width / 2 for p in x])
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    plt.show()


if __name__ == "__main__":

    text = read_text_from_docx('Test.docx')

    search_words = ["Відьмак", "Геральт", "монстр", "CD Projekt RED", "магія"]

    # Замір часу для безпосереднього пошуку
    start_time = time.time()
    naive_results = naive_search(text, search_words)
    naive_search_time = time.time() - start_time
    naive_counts = {word: len(occurrences) for word, occurrences in naive_results.items()}
    print("Результати безпосереднього пошуку:", naive_counts)
    print("Час безпосереднього пошуку:", naive_search_time, "секунд")

    # Замір часу для Ахо-Корасік
    start_time = time.time()
    aho_corasick = AhoCorasick()
    for word in search_words:
        aho_corasick.add_word(word)
    aho_corasick.build_failure_pointers()
    aho_corasick_results = aho_corasick.search(text)
    aho_corasick_search_time = time.time() - start_time
    aho_counts = {word: len(positions) for word, positions in aho_corasick_results.items()}
    print("Результати пошуку Ахо-Корасік:", aho_counts)
    print("Час пошуку Ахо-Корасік:", aho_corasick_search_time, "секунд")

    # Візуалізація результатів
    plot_search_results(naive_counts, aho_counts)

    # Два рядки для аналізу відстані Левенштейна-Дамерау
    line1 = "Секрет успіху ігор криється не лише в популярності книг про Відьмака, адже навіть не всі вони були перекладені англійською. " \
            "Розробникам із CD Projekt RED вже у першій грі «Відьмак», випущеній у 2007 році, вдалося передати унікальний дух творів Сапковського."
    line2 = "Те незвичайне, що найбільше подобається фанатам у книгах та іграх про Відьмака, це сам герой – Геральт із Рівії, відьмак. «Відьмак» – це персонаж слов'янської міфології. " \
            "Саме слово походить від праслов'янського * vede («знати»)."
    distance, operations = damerau_levenshtein_distance_with_path(line1, line2)
    print("Відстань Левенштейна-Дамерау:", distance)
    print("Операції для перетворення рядка:")
    for op in operations:
        print(op)

