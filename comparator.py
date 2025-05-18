

import argparse
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
import algo3
import dynamic



import matplotlib.pyplot as plt
from collections import Counter

from collections import Counter
import matplotlib.pyplot as plt

def stats(path_1, path_2):
    count_tokens_gold = 0
    count_tokens_second = 0
    count_good = 0
    mots_bons = 0

    len_gold = Counter()
    len_second = Counter()

    with open(path_1, "r", encoding="utf-8") as gold, open(path_2, "r", encoding="utf-8") as second:
        lines_gold = gold.readlines()
        lines_second = second.readlines()

        for line_gold, line_second in zip(lines_gold, lines_second):
            tab_gold = line_gold.strip().split()
            tab_second = line_second.strip().split()

            count_tokens_gold += len(tab_gold)
            count_tokens_second += len(tab_second)

            for token in tab_gold:
                len_gold[len(token)] += 1
            for token in tab_second:
                len_second[len(token)] += 1

            # Parcours intelligent sans chevauchement
            idx_second = 0
            success = True
            for token_gold in tab_gold:
                target = token_gold
                composed = ""
                while len(composed) < len(target) and idx_second < len(tab_second):
                    composed += tab_second[idx_second]
                    idx_second += 1
                if composed != target:
                    success = False
                    break

            if success and idx_second == len(tab_second):  # Tous les tokens utilisés, dans l'ordre
                count_good += len(tab_gold)
                mots_bons+=1
            else:
                print(f"mots tokenisés différemment : {tab_gold} / {tab_second}")

    print(f"Nombre total de tokens - 1ère version : {count_tokens_gold}")
    print(f" Nombre total de tokens - 2e version   : {count_tokens_second}")
    print(f"Nombre de mots correctement tokenisés: {mots_bons}")
    print(f" Nombre de tokens correpondants : {count_good}")

    all_lengths = sorted(set(len_gold.keys()) | set(len_second.keys()))
    plt.bar(all_lengths, [len_gold.get(k, 0) for k in all_lengths], width=0.4, label="Tokens1", align="edge", alpha=0.7)
    plt.bar(all_lengths, [len_second.get(k, 0) for k in all_lengths], width=-0.4, label="Tokens2", align="edge", alpha=0.7)

    plt.legend()
    plt.title("Distribution de la taille des tokens")
    plt.xlabel("Taille (nb de caractères)")
    plt.ylabel("Fréquence")
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_text1',help="path to the reference tokenized text with the format 'token1 token2' and a word per line")
    parser.add_argument('path_to_text2',help="path to the tokenized text tested with the format 'token1 token2' and a word per line, it must have the same word at each line that the reference text")
    args = parser.parse_args()
    stats(args.path_to_text1,args.path_to_text2)



