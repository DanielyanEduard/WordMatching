from flask import Flask, render_template, request

from gensim.models import KeyedVectors, Word2Vec
from gensim import downloader as api  # Correct import

import re
from typing import List, Union
import random

app = Flask(__name__)

# dataset = api.load("text8")

# Train the Word2Vec model using the text8 corpus
# model = Word2Vec(sentences=dataset, vector_size=100, window=5, min_count=5, workers=4)

# Save the trained model (optional)
# model.save("text8_word2vec.model")
# model_path = "bin/text8_word2vec.bin"
# model = api.load("text8")
# model.wv.most_similar("bin/text8_word2vec.bin", binary=True)
model = Word2Vec.load("text8_word2vec.model")
word_vectors = model.wv#KeyedVectors.load_word2vec_format(model_path, binary=True)


def get_related_words(word: str = None, top_n: int = 30, phrase: str = None) -> Union[List[str], bool]:
    """
        Retrieves the most similar words to the given word using a pre-trained Word2Vec model.
        Filters out inflected forms of the word.

        :param word: The input word to find related words for.
        :param top_n: Number of similar words to retrieve.
        :param phrase: The phrase to search for.
        :return: A list of related words or False if the word is not found in the model.
    """
    if word is None:
        word = random.choice(word_vectors.index_to_key[:500])
        top_n = len(word) * 3
    def are_inflected_forms(word1, word2, phrase: str = None):
        """Checks if two words are inflected forms of each other or if a word is in the wrong format."""
        if not word2.isalpha() or word2.lower() in phrase.lower() or any(word in word2.lower() for word in phrase.lower().split()):
            return True

        return re.sub(r"(ing|er|est|s|es|ly|ness|ment|ed)$", "", word1)[:-1] == \
               re.sub(r"(ing|er|est|s|es|ly|ness|ment|ed)$", "", word2)[:-1]
    try:
        related_words = word_vectors.most_similar(word, topn=top_n)
        return [w for w, _ in related_words if not are_inflected_forms(word, w, phrase)]
    except KeyError:
        word = random.choice(word_vectors.index_to_key[:500])
        print("That word is not in the vocabulary. Used", word, "instead.")
        return get_related_words(word, len(word) * 3)


def find_best_match(phrase: str, words: List[str]) -> Union[str, bool]:
    """
        Matches words to a phrase based on letter sequence similarity.

        :param phrase: The target phrase to match.
        :param words: A list of words to match against the phrase.
        :return: A formatted string with best-matching words or False if no match is found.
    """
    def format_word(word: str, letters: str) -> str:
        """Formats the matched word by capitalizing the matched letters."""
        i = 0
        result = []
        while i < len(word):
            if len(letters) > 0 and word[i] == letters[0]:
                result.append(word[i].upper())
                letters = letters[1:]
                i += 1
                if i < len(word):
                    result.append(word[i].lower())
            else:
                result.append(word[i].lower())
            i += 1
        return "".join(result)

    phrase = phrase.replace(" ", "").lower()
    selected_words = []
    i = 0
    while i < len(phrase):
        counts = []
        for word in words:
            edited_word = word[:].lower()
            j = i
            count = 0
            while j < len(phrase) and phrase[j] in edited_word.lower():
                count += 1
                j += 1
                try:
                    edited_word = edited_word[edited_word.index(phrase[j-1])+2:]
                except ValueError:
                    break
            counts.append(count)
        if max(counts) == 0:
            return False
        max_match = max(counts)
        best_word = words.pop(counts.index(max_match))
        selected_words.append(format_word(best_word.lower(), phrase[i:i + max_match]))
        i += max_match
    return " ".join(selected_words) if i == len(phrase) else False


def get_words_by_phrase_and_subject(phrase: str, subject: str = None) -> str:
    """
        Generates words matching the given phrase based on related words from a subject.

        :param phrase: The phrase to match words against.
        :param subject: The subject word to find related words from.
        :return: A formatted string of matched words.
    """
    if subject == "":
        subject = random.choice(word_vectors.index_to_key[:500])
    top_n = len(phrase) * 3
    while not (matched := find_best_match(phrase, get_related_words(subject, top_n, phrase))):
        top_n *= 2
    return matched

@app.route("/", methods=["GET", "POST"])
def home():
    output = None
    if request.method == "POST":
        input1 = request.form["input1"]
        input2 = request.form.get("input2", "")  # Optional input
        output = get_words_by_phrase_and_subject(input1, input2)
    return render_template("index.html", output=output)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
