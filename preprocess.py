from nltk.corpus import reuters
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk


def Tokenization(text):
    custom_tokenizer = nltk.RegexpTokenizer(r"\w+(?:[-']\w+)*|\d+(?:\.\d+)?|'[^']+'|\"[^\"]+\"|\S")
    tokens = custom_tokenizer.tokenize(text)
    return tokens


def SentenceSplitting(text):
    custom_sent_tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
    sentences = custom_sent_tokenizer.tokenize(text)
    return sentences


def POSTaagging(text):
    tokens = Tokenization(text)
    pos_tags = pos_tag(tokens)
    return pos_tags


def GazetteerAnnotation(text):
    tokens = Tokenization(text)

    country_gazetteer = {
        'Canada', 'Denmark', 'England', 'France', 'Germany', 'Hungary', 'Indonesia', 'Philippines', 'JAPAN', 'U.S.', 'U.K.'
    }
    unit_gazetteer = {
        'kg', 'cm', 'm', 'ft', 'g', 'lb', 'tonnes',
    }
    year_gazetteer = {str(year) for year in range(1980, 2024)}

    crop_gazetteer = {
        'wheat', 'corn', 'rice', 'soybean', 'cotton', 'copra', 'Sunflowerseed',
    }

    gazetteer_names = {
        'country': country_gazetteer,
        'unit': unit_gazetteer,
        'year': year_gazetteer,
        'crop': crop_gazetteer
    }

    annotated_tokens = []

    for token in tokens:
        annotation = 'O'
        for gazetteer_name, gazetteer_list in gazetteer_names.items():
            if token in gazetteer_list:
                annotation = gazetteer_name
                break

        annotated_tokens.append((token, annotation))

    return annotated_tokens, gazetteer_names


def ner_with_gazetteer_entities(text, gazetteers):
    ne_tree = ne_chunk(pos_tag(word_tokenize(text)), binary=True)

    annotated_entities = []

    for subtree in ne_tree:
        if type(subtree) == nltk.Tree:
            entity_name = " ".join([token for token, _ in subtree.leaves()])
            for gazetteer_name, gazetteer_list in gazetteers.items():
                if entity_name in gazetteer_list:
                    annotated_entities.append((entity_name, gazetteer_name))
                    break
        else:
            token, pos_tag_ = subtree
            if token.lower() in [item.lower() for item in gazetteers['year']]:
                annotated_entities.append((token, 'year'))
            elif token.lower() in [item.lower() for item in gazetteers['crop']]:
                annotated_entities.append((token, 'crop'))
            elif token.lower() in [item.lower() for item in gazetteers['country']]:
                annotated_entities.append((token, 'country'))
            elif token.lower() in [item.lower() for item in gazetteers['unit']]:
                annotated_entities.append((token, 'unit'))

    return set(annotated_entities)


def extract_measured_entities(text):
    units = ['cm', 'kg', 'Ecus', 'tonnes', 'pct', 'billion dollars', 'mln tonnes', 'mln stg', 'billion']
    unit_pattern = '|'.join(re.escape(unit) for unit in units)
    pattern = rf'(\d+(?:,\d{{3}})*(?:\.\d+)?\s*(?:{unit_pattern})\b)'

    matches = re.findall(pattern, text, re.IGNORECASE)

    return set(matches)


def preprocess(document_id):
    text = reuters.raw(document_id)

    tokens = Tokenization(text)

    sentences = SentenceSplitting(text)

    pos_tags = POSTaagging(text)

    annotated_tokens, gazetteers = GazetteerAnnotation(text)

    ne_tree = ner_with_gazetteer_entities(text, gazetteers)

    ms_entity = extract_measured_entities(text)

    return sentences, tokens, pos_tags, annotated_tokens, ne_tree, ms_entity


if __name__ == "__main__":
    document_id = 'training/9880'
    sentences, tokens, pos_tags, annotated_tokens, ne_tree, ms_entity = preprocess(document_id)

    print("Sentences:", sentences)
    print("\nTokens:", tokens)
    print("\nPOS Tags:", pos_tags)
    print("\nAnnotated Tokens:", annotated_tokens)
    print("\nNamed Entity Recognition (NER)  :", ne_tree)
    print("\nMeasured Entity Detection:", ms_entity)