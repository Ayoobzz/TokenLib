import argparse
import random
import os
import sys

def read_data(path):
    data = list()
    try:
        with open(path, 'r', encoding='utf-8') as instream:
            cols = next(instream).strip().split(",")
            for line in instream:
                data.append({
                    k: v for k, v in zip(cols, line.strip().split(","))
                })
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        sys.exit(1)

def read_data_dict(path):
    data = dict()
    try:
        with open(path, 'r', encoding='utf-8') as instream:
            cols = next(instream).strip().split(",")
            key_field = cols[0]
            for line in instream:
                values = line.strip().split(",")
                if len(values) != len(cols):
                    print(f"Warning: Skipping line due to mismatch in columns: {line.strip()}")
                    continue
                entry = {k: v for k, v in zip(cols, values)}
                data[entry[key_field]] = entry
        return data
    except FileNotFoundError:
        print(f"Error: File not found: {path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        sys.exit(1)

def tokenize_text(text_path, output_path, data):
    if not os.path.exists(text_path):
        print(f"Error: Text file does not exist: {text_path}")
        sys.exit(1)

    print("Tokenizing text...")
    try:
        with open(text_path, 'r', encoding="utf-8") as texte:
            lines = texte.readlines()
            with open(output_path, 'w', encoding="utf-8") as output:
                for line in lines:
                    key = line.strip()
                    if key in data:
                        to_write = data[key]["segmentation"].replace('-', ' ')
                        output.write(to_write + "\n")
                    else:
                        print(f"Warning: Word not found in data: '{key}'")
        print(f"Tokenization completed. Output written to {output_path}")
    except Exception as e:
        print(f"Error during tokenization: {e}")
        sys.exit(1)

def tokenize_generator(output_path, data, num_words=1000, lang=None):
    print("Generating tokenized text...")
    if lang is not None:
        langs = lang.split()
        data_lang = [word for word in data if word["lang"] in langs]
        print(f"Filtering on languages: {langs}")
    else:
        data_lang = data

    if not data_lang:
        print("Error: No data available for the selected language(s).")
        sys.exit(1)

    if len(data_lang) < num_words:
        print(f"Warning: Only {len(data_lang)} entries available, using all of them instead of requested {num_words}.")
        final_data = data_lang
    else:
        final_data = random.sample(data_lang, num_words)

    try:
        with open(output_path, 'w', encoding="utf-8") as output:
            for elem in final_data:
                output.write(elem["segmentation"].replace('-', ' ') + "\n")
        print(f"Generated text written to {output_path}")
    except Exception as e:
        print(f"Error writing to file {output_path}: {e}")
        sys.exit(1)

def launcher(fun, output, text=None, num=None, lang=None):
    if fun == '1':
        if not text:
            print("Error: Text file path (-t) is required for function 1.")
            sys.exit(1)
        data = read_data_dict('train.csv')
        tokenize_text(text, output, data)

    elif fun == '2':
        if num: 
            try:
                num_int = int(num)
            except ValueError:
                print(f"Error: Provided number of words (-n) is not an integer: '{num}'")
                sys.exit(1)
        data = read_data('train.csv')
        tokenize_generator(output, data, num_int, lang)

    else:
        print(f"Error: Unknown function '{fun}'. Choose '1' or '2'.")
        sys.exit(1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fun', help="Fonction souhaitée: '1' pour tokenizer un texte avec la segmentation morphologique, '2' pour en générer un.")
    parser.add_argument('output_path', help="Nom sous lequel le fichier de sortie sera enregistré.")
    parser.add_argument('-t', default=None, help="(Requis pour fun 1) Chemin vers le texte à tokenizer.")
    parser.add_argument('-n', default=None, help="(Optionnel) Nombre de mots générés pour fun 2.")
    parser.add_argument('-l', default=None, help="(Optionnel) Langues dans lesquelles générer le texte, ex: 'fr en de'")
    args = parser.parse_args()
    launcher(args.fun, args.output_path, args.t, args.n, args.l)
