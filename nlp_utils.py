import re
import nltk
import spacy
import langid
import random
import requests
from langid.langid import classify
from difflib import SequenceMatcher
from nltk import pos_tag
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from google.generativeai import GenerativeModel, configure

configure(api_key="AIzaSyAFsbEx5fpzsoePD_1Wyct63A8PgBejBKI")

# Then initialize your model without the api_key argument
model = GenerativeModel('gemini-pro')


# Initialize NLP models and tools
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")

import random
import openai #for generation with Mistral
from google.api_core.exceptions import ResourceExhausted #to note if Gemini quota is exhausted and use Mistral instead

openai.api_key = "" 

# Function definitions
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return 'adjective'
    elif tag.startswith('V'):
        return 'verb'
    elif tag.startswith('N'):
        return 'noun'
    elif tag.startswith('R'):
        return 'adverb'
    else:
        return 'noun'  # Default to noun if the POS tag is not recognized

def find_target_word(context, vocabulary_item):
    # Tokenize the context into words
    words = re.findall(r'\w+', context.lower())
    vocabulary_item_lower = vocabulary_item.lower()

    # Check if the vocabulary item is in the context
    if vocabulary_item_lower in words:
        return vocabulary_item

    # Check for lemma match
    for word in words:
        if lemmatizer.lemmatize(word) == vocabulary_item_lower:
            return word

    # Find the word that shares the most letters with the vocabulary item
    most_similar_word = max(words, key=lambda word: SequenceMatcher(None, word, vocabulary_item_lower).ratio())
    return most_similar_word


def lemmatize_sentence(sentence):
    doc = nlp(sentence)
    lemmatized_words = [token.lemma_ for token in doc]
    lemmatized_sentence = ' '.join(lemmatized_words)
    return lemmatized_sentence


def validate_context(context, item, pos, attempts_counter):
    # Initial checks
    if not context or len(context.split()) <= 4:
        return False, attempts_counter

    context = context.strip()
    item_lower = item.lower()
    lemmatized_context = lemmatize_sentence(context).lower()

    # Check if the item or its lemma is present in the sentence
    if not (item_lower in lemmatized_context or item_lower in context or lemmatize_sentence(item_lower) in lemmatized_context):
        return False, attempts_counter

    # Remove unwanted patterns
    context = re.sub(r'^\d+(\.|\))', '', context)
    context = re.sub(r'^[^\w\s]+(?=[A-Z])', '', context)
    context = re.sub(r'Here is a sentence.*?:', '', context)

    # Further cleaning
    sentences = re.split(r'(?<=[.!?]) +', context)
    if len(sentences) > 1:
        if sentences[-1].startswith('In this sentence') or sentences[-1].startswith('In the sentence'):
            sentences.pop()
        context = ' '.join(sentences)

    # Remove unwanted characters
    context = re.sub(r'^[\'\"“‘]+|[\'\"”’]+$', '', context)
    context = re.sub(r'\.[\'"”’]+\.$', '.', context)
    context = context.strip()

    # POS tag validation
    if pos in ["noun", "verb", "adjective", "adverb"]:
        tokens = word_tokenize(context)
        pos_tags = pos_tag(tokens)

        pos_compatible = False
        for word, word_pos in pos_tags:
            if word.lower() == item_lower:
                pos_compatible = ((pos == 'noun' and word_pos in ['NN', 'NNS', 'NNP']) or
                                  (pos == 'verb' and word_pos in ['VB', 'VBP', 'VBG', 'VBD', 'VBZ', 'VBN']) or
                                  (pos == 'adjective' and word_pos in ['JJ', 'VBG', 'VBD']) or
                                  (pos == 'adverb' and word_pos == 'RB'))
                if not pos_compatible:
                    return False, attempts_counter
                break

    # Language validation
    lang, confidence = classify(context)
    if lang != 'en':
        return False, attempts_counter

    # Ensure item is used only once
    words = word_tokenize(context.lower())
    occurrences_base_form = words.count(item_lower)
    occurrences_base_lemma = lemmatized_context.count(lemmatize_sentence(item_lower))
    if occurrences_base_form > 1 or occurrences_base_lemma > 1:
        return False, attempts_counter

    # If all checks pass, return True
    return True, attempts_counter


def check_linguistic_features(example, level):
    # Tokenize the example into sentences and words
    sentences = sent_tokenize(example)
    words = [word for word in word_tokenize(example) if word.isalpha()]

    # If words list is empty, return an empty string as it doesn't pass the validation
    if not words or not sentences:
        return ""

    # Initialize a flag to track if all criteria are met
    all_within_range = True

    # 1. Words per sentence
    words_per_sentence = len(words) / len(sentences)
    print(f"Words per sentence: {words_per_sentence}")
    if level == 'B1':
        if (words_per_sentence >= 8) and (words_per_sentence <= 26.05): 
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if (words_per_sentence >= 17) and (words_per_sentence <= 33.69):
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 2. Letters per word
    letters_per_word = sum(len(word) for word in words) / len(words)
    print(f"Letters per word: {letters_per_word}")
    if level == 'B1':
        if letters_per_word <= 9.26:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if letters_per_word <= 10.96:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 3. Noun phrases per sentence
    total_noun_phrases = sum(len(list(nlp(sent).noun_chunks)) for sent in sentences)
    noun_phrases_per_sentence = total_noun_phrases / len(sentences)
    print(f"Noun phrases per sentence: {noun_phrases_per_sentence}")
    if level == 'B1':
        if noun_phrases_per_sentence <= 10.49:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if (noun_phrases_per_sentence >= 0.47) and (noun_phrases_per_sentence <= 11.71):
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 4. Non-stem words per sentence
    stemmer = PorterStemmer()
    total_non_stem_words = sum(1 for word in words if word.lower() != stemmer.stem(word.lower()))
    non_stem_words_per_sentence = total_non_stem_words / len(sentences)
    print(f"Non-stem words per sentence: {non_stem_words_per_sentence}")
    if level == 'B1':
        if (non_stem_words_per_sentence >= 2.47) and (non_stem_words_per_sentence <= 55.52):
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if non_stem_words_per_sentence <= 54.46:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 5. Punctuation signs per sentence
    punctuation_to_count = "',:;-\"‘’“”"
    total_punctuation = sum(example.count(char) for char in punctuation_to_count)
    punctuation_per_sentence = total_punctuation / len(sentences)
    print(f"Punctuation signs per sentence: {punctuation_per_sentence}")
    if level == 'B1':
        if punctuation_per_sentence <= 2.41:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if punctuation_per_sentence <= 2.82:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 6. Verbs per sentence
    total_verbs = sum(1 for token in nlp(example) if token.pos_ == 'VERB')
    verbs_per_sentence = total_verbs / len(sentences)
    print(f"Verbs per sentence: {verbs_per_sentence}")
    if level == 'B1':
        if verbs_per_sentence <= 4.95:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if (verbs_per_sentence >= 0.45) and (verbs_per_sentence <= 4.77):
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 7. Adjectives and adverbs per sentence
    total_adj_adv = sum(1 for token in nlp(example) if token.pos_ in ('ADJ', 'ADV'))
    adj_adv_per_sentence = total_adj_adv / len(sentences)
    print(f"Adjectives and adverbs per sentence: {adj_adv_per_sentence}")
    if level == 'B1':
        if adj_adv_per_sentence <= 6.4:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if (adj_adv_per_sentence >= 0.99) and (adj_adv_per_sentence <= 5.15):
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 8. First person pronouns per sentence
    first_person_pronouns = {"i", "me", "my", "myself", "we", "us", "our", "ours", "ourselves"}
    total_first_person_pronouns = sum(1 for word in words if word.lower() in first_person_pronouns)
    first_person_pronouns_per_sentence = total_first_person_pronouns / len(sentences)
    print(f"First person pronouns per sentence: {first_person_pronouns_per_sentence}")
    if level == 'B1':
        if first_person_pronouns_per_sentence <= 1:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if first_person_pronouns_per_sentence <= 0.79:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 9. Proper nouns per sentence
    total_proper_nouns = sum(1 for token in nlp(example) if token.pos_ == 'PROPN')
    proper_nouns_per_sentence = total_proper_nouns / len(sentences)
    print(f"Proper nouns per sentence: {proper_nouns_per_sentence}")
    if level == 'B1':
        if proper_nouns_per_sentence <= 3.39:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if proper_nouns_per_sentence <= 5.19:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 10. Pronouns per sentence
    total_pronouns = sum(1 for token in nlp(example) if token.pos_ == 'PRON')
    pronouns_per_sentence = total_pronouns / len(sentences)
    print(f"Pronouns per sentence: {pronouns_per_sentence}")
    if level == 'B1':
        if pronouns_per_sentence <= 2.65:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if pronouns_per_sentence <= 3.11:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # 11. Anaphora words per sentence
    anaphora_words = set(["the", "he", "she", "it", "they", "this", "that", "these", "those",
                          "who", "which", "whose", "whom", "where", "all",
                          "some", "none", "any", "each", "every", "here", "there", "now", "then"])
    total_anaphora_words = sum(1 for word in word_tokenize(example) if word.lower() in anaphora_words)
    anaphora_words_per_sentence = (total_anaphora_words / len(words)) * 100
    print(f"Anaphora words per sentence: {anaphora_words_per_sentence}")
    if level == 'B1':
        if (anaphora_words_per_sentence >= 0.13) and (anaphora_words_per_sentence <= 25.77):
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False
    if level == 'B2':
        if anaphora_words_per_sentence <= 23.7:
            print("Falls within the range")
        else:
            print("Doesn't fall within the range")
            all_within_range = False

    # Set example to empty string if not all criteria are met
    if not all_within_range:
        example = ""

    return example




def clean_text(sentences):
    cleaned_sentences = []
    for text in sentences:
        text = text.replace(' ,', ',')
        text = text.replace(' .', '.')
        text = text.replace('\n', ' ')
        text = text.lstrip('#')
        text = text.lstrip('-')
        text = text.lstrip('•')
        text = text.lstrip('>')
        text = text.replace('*', '')

        # If text starts with up to 3 words followed by ':', remove this part:
        parts = text.split(':', 1)
        if len(parts) > 1 and len(parts[0].split()) <= 3:
            text = parts[1].strip()

        # If the sentence now ends in ':.' remove this pattern as well as anything before it following the last end-of-sentence punctuation
        if text.endswith(':.'):
            text = text.replace(':.', '')
            if ('.' in text) or ('!' in text) or ('?' in text):
                last_punctuation_index = max(text.rfind('.'), text.rfind('?'), text.rfind('!'))
                text = text[:last_punctuation_index].strip()

        # Remove text within square brackets at the beginning of the text
        text = re.sub(r'^\[[^\]]]*\]\s*', '', text)

        if 'Fig.' in text:
            open_bracket_index = text.find('(')
            if open_bracket_index != -1 and text.count('(') > text.count(')'):
                text = text[:open_bracket_index].strip()

        # Remove opening quotation sign if not closed
        if text.startswith(("'", '"', '“', '‘', '„', '‹', '«')):
            if text.endswith(("'", '"', '”', '’', '”', '›', '»')) and text[0] != text[-1]:
                text = text[1:]
            elif text.startswith(("'", '"')) and not text.endswith(("'", '"')):
                text = text[1:]

        # Remove quotation marks if the sentence starts with one and ends with a period, followed by a closing quotation mark
        if text.startswith(("'", '"', '“', '‘', '„', '‹', '«')) and text.endswith((".'", '."', '.”', '.’')):
            text = text[1:-2].strip() + '.'

        # Capitalize the first letter if it's not already capitalized
        if text and not text[0].isupper():
            text = text[0].upper() + text[1:]

        # Remove short text in beginning like enumeration
        match_short_text = re.match(r'^[a-zA-Z0-9\s]{1,10}\.\s*', text)
        if match_short_text:
            text = text[len(match_short_text.group()):]

        # Put a full stop if no punctuation at the end 
        if not text.endswith(('.', '!', '?')):
            text += '.'

        # Strip extra spaces
        text = text.strip()

        # Remove quotations if the entire sentence is within them
        if text.startswith(("'", '"', '“', '‘', '„', '‹', '«')) and text.endswith(("'", '"', '”', '’', '”', '›', '»')):
            text = text[1:-1].strip()

        cleaned_sentences.append(text)
    return cleaned_sentences



def generate_and_validate_contexts(item, pos, domain, level):
    contexts = []
    lower = 8 if level == 'B1' else 17
    upper = 43 if level == 'B1' else 87

    question_template = "Here is a sentence at level {level} showing how you use the {pos} '{item}' in the domain of {domain} ({lower}-{upper} words):"
    question = question_template.format(
        level=level,
        pos=pos,
        item=item,
        domain=domain,
        lower=lower,
        upper=upper
    )

    models = ['gemini', 'mistral']
    random.shuffle(models)  # Randomly choose the order of models

    attempts_counter = 0  # Initialize attempts counter
    linguistic_checks_enabled = True  # Enable linguistic checks initially

    while len(contexts) < 5:  # Ensure 5 valid contexts are generated
        for model_name in models:
            try:
                if model_name == 'gemini':
                    response = model.generate_content(question)
                    context = response.text if response else ''
                    generation_method = 'Gemini'
                elif model_name == 'mistral':
                    response = requests.post(
                        "http://localhost:1234/v1/chat/completions",
                        json={
                            "model": "local-model",
                            "messages": [
                                {"role": "system", "content": question}
                            ],
                            "temperature": 0.8
                        }
                    )
                    if response.status_code == 200:
                        response_json = response.json()
                        context = response_json["choices"][0]["message"]["content"]
                        generation_method = 'Mistral'
                    else:
                        print(f"Error generating context with Mistral: {response.status_code} {response.text}")
                        context = ''
                
                context = context.strip()
                is_valid, attempts_counter = validate_context(context, item, pos, attempts_counter)
                
                if is_valid and context:
                    # Perform linguistic feature checks if enabled
                    if linguistic_checks_enabled:
                        context = check_linguistic_features(context, level)
                        if context:
                            print(f"Generated with {generation_method}: {context}")  # Output the method to the console
                            contexts.append((context, generation_method))
                        else:
                            attempts_counter += 1  # Increment the attempts counter if the check fails
                    else:
                        print(f"Generated with {generation_method}: {context}")  # Output the method to the console
                        contexts.append((context, generation_method))
                    
                    # Disable linguistic checks if 30 attempts have been made
                    if attempts_counter >= 30:
                        linguistic_checks_enabled = False

                if len(contexts) >= 5 or attempts_counter >= 100:
                    break
            except Exception as e:
                print(f"Error generating context with {model_name}: {e}")
                continue  # Try the next model if this one fails

    cleaned_contexts = clean_text([c[0] for c in contexts])  # Clean the contexts
    return [(cleaned_contexts[i], contexts[i][1]) for i in range(len(contexts))]  # Return cleaned contexts with methods






def strip_prefixes(item):
    prefixes = ['to ', 'a ', 'an ', 'the ']
    for prefix in prefixes:
        if item.lower().startswith(prefix):
            return item[len(prefix):]
    return item


def update_pos(app):
    with app.app_context():
        entries = VocabularyEntry.query.all()
        for entry in entries:
            if not entry.pos or entry.pos == 'word':
                if ' ' in entry.item:
                    entry.pos = 'expression'
                else:
                    nltk_tag = pos_tag([entry.item])[0][1]
                    entry.pos = get_wordnet_pos(nltk_tag)
        db.session.commit()

