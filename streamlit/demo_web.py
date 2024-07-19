import streamlit as st
import numpy as np
import gensim
from gensim.models import Word2Vec
import editdistance

# Sample phonetic encoding map for demonstration
phonetic_map = {
    'क': '1', 'ख': '1', 'ग': '1', 'घ': '1',
    'च': '2', 'छ': '2', 'ज': '2', 'झ': '2',
    'ट': '3', 'ठ': '3', 'ड': '3', 'ढ': '3',
    'त': '4', 'थ': '4', 'द': '4', 'ध': '4',
    'प': '5', 'फ': '5', 'ब': '5', 'भ': '5',
    'य': '6', 'र': '6', 'ल': '6', 'व': '6',
    'श': '7', 'ष': '7', 'स': '7', 'ह': '7'
}

def phonetic_encode(word):
    return ''.join(phonetic_map.get(char, '0') for char in word)

# Sample transliteration map
transliteration_map = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo', 'ऋ': 'ri', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au',
    'अं': 'am', 'अः': 'ah', 'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo', 'ृ': 'ri', 'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au',
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga', 'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na', 'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma', 'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'va', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha', 'क्ष': 'kshya', 'त्र': 'tra', 'ज्ञ': 'gya', 'ऽ': "'", 'ँ': 'n', 'ं': 'm', 'ः': 'h',
    '़': '', '्': '', '०': '0', '१': '1', '२': '2', '३': '3', '४': '4', '५': '5', '६': '6', '७': '7', '८': '8', '९': '9',
    'ॐ': 'om', 'क़': 'qa', 'ख़': 'kha', 'ग़': 'gha', 'ज़': 'za', 'ड़': 'ra', 'ढ़': 'rha', 'फ़': 'fa', 'य़': 'ya'
}

def transliterate(word):
    transliterated = ''.join(transliteration_map.get(char, char) for char in word)
    # transliterated = word
    return transliterated

# Generate the possible candidates word using the edit distance
def generate_candidates(word, vocabulary, max_distance=1):
    candidates = []
    transliterated_word = transliterate(word)
    phonetic_word = phonetic_encode(word)
    first_char = transliterated_word[0]  # Get the first character of the transliterated word
    last_char = transliterated_word[-1]
    for vocab_word in vocabulary:
        # Skip if the vocab_word is the same as the input word
        if vocab_word == transliterated_word:
            continue
        if phonetic_encode(vocab_word) == phonetic_word:
            transliterated_vocab_word = transliterate(vocab_word)
            if transliterated_vocab_word.startswith(first_char) and transliterated_vocab_word.endswith(last_char):  # Check if first character matches
                if len(transliterated_vocab_word) == len(transliterated_word):
                    distance = editdistance.eval(transliterated_word, transliterated_vocab_word)
                    if distance <= max_distance:
                        candidates.append((vocab_word, distance))
    return sorted(candidates, key=lambda x: x[1])

# Based on the word embeddings on our word2vec model, filter out the candidates word based on the sentence
def filter_by_context(candidates, context_words, model):
    filtered_candidates = []
    for candidate, distance in candidates:
        candidate_score = 0
        for context_word in context_words:
            if context_word in model.wv and candidate in model.wv:
                candidate_score += model.wv.similarity(context_word, candidate)
        filtered_candidates.append((candidate, distance, candidate_score))

    # Sort by score (descending) and distance (ascending)
    filtered_candidates.sort(key=lambda x: (-x[2], x[1]))
    return filtered_candidates if filtered_candidates else candidates

# Computing the similarity score between one word and their context words in the same sentence
def calculate_similarity_scores(words, model, filtered_whole_vocab):
    context_similarities = {}
    for i, word in enumerate(words):
        if word in filtered_whole_vocab:
            context_words = [w for j, w in enumerate(words) if j != i and w in model.wv]
            if context_words:
                similarity_scores = [model.wv.similarity(word, context_word) for context_word in context_words]
                context_similarities[word] = np.mean(similarity_scores)
            else:
                context_similarities[word] = 0
        else:
            context_similarities[word] = 0  # Word not in the model
    return context_similarities

def correct_sentence(sentence, vocabulary, model, threshold=0.4):
    words = sentence.split()
    corrected_sentence = []
    context_similarities = calculate_similarity_scores(words, model, vocabulary)

    for i, word in enumerate(words):
        if context_similarities.get(word, 0) < threshold and context_similarities.get(word, 0) != 0:
            # Generate candidates for the word
            candidates = generate_candidates(word, vocabulary)
            # st.write(candidates)
            # Filter candidates based on context
            filtered_candidates = filter_by_context(candidates, words, model)
            st.write("filtered candidates:",filtered_candidates)
            # Select the best candidate (here, the first one after filtering)
            best_candidate = filtered_candidates[0][0] if filtered_candidates else word
            corrected_sentence.append(best_candidate)
        else:
            corrected_sentence.append(word)  # If word is in vocabulary and above threshold, use it as is

    return ' '.join(corrected_sentence), context_similarities


# Load your vocabulary and model
with open("C:/Users/nabin shrestha/Downloads/filtered_nostop_check_mate_model_vocabulary.txt",'r',encoding='utf-8') as f:
    filtered_whole_vocab = f.read().splitlines()
# vocabulary = list(vocab)
check_mate_model = gensim.models.Word2Vec.load("C:/Users/nabin shrestha/Downloads/checked_mate_word2vec_model.model")  # load Word2Vec model

st.title("Nepali Contextual Spell Checker")
def main():
    # Sentence correction section
    # st.header("Correct a Sentence")
    sentence = st.text_input("Enter a sentence in Nepali:", "")
    if sentence:
        corrected_sentence, context_similarities = correct_sentence(sentence, filtered_whole_vocab, check_mate_model,threshold=0.35)
        # st.write("Corrected Sentence:", corrected_sentence)
        st.write("Context Similarities:", context_similarities)

    # Similar words section
    st.header("Find Similar Words")
    word = st.text_input("Enter a word in Nepali:", "")
    if word:
        similar_words = check_mate_model.wv.most_similar(word)
        st.write(f"Words similar to '{word}':")
        for similar_word, similarity in similar_words:
            st.write(f"{similar_word}: {similarity:.4f}")

if __name__ == "__main__":
    main()
