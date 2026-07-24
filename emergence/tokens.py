"""Concept-word set: 12 semantic categories x 10 words.

Each word is used with a leading space and kept only if it maps to a single
token under the model tokenizer, giving one vector per word.
"""

CANDIDATE_WORDS = [
    # animals
    "dog", "cat", "horse", "fish", "bird", "lion", "bear", "wolf", "mouse", "snake",
    # colors
    "red", "blue", "green", "yellow", "black", "white", "brown", "pink", "purple", "gray",
    # numbers
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    # body
    "hand", "head", "eye", "heart", "foot", "arm", "face", "blood", "bone", "skin",
    # nature
    "tree", "river", "mountain", "ocean", "rain", "snow", "sun", "moon", "star", "wind",
    # food
    "bread", "milk", "meat", "fruit", "wine", "sugar", "salt", "rice", "tea", "coffee",
    # objects
    "table", "chair", "door", "window", "car", "boat", "knife", "clock", "phone", "book",
    # people / roles
    "king", "queen", "doctor", "teacher", "soldier", "mother", "father", "child", "friend", "president",
    # places
    "city", "country", "school", "church", "hospital", "street", "market", "island", "bridge", "house",
    # abstract
    "love", "death", "time", "war", "peace", "money", "power", "music", "truth", "dream",
    # adjectives
    "big", "small", "hot", "cold", "old", "new", "fast", "slow", "strong", "weak",
    # weather / time
    "winter", "summer", "morning", "night", "week", "year", "fire", "ice", "gold", "iron",
]


def select_single_token_words(tokenizer, words=None):
    """Return (words, token_ids) for words that are a single token with a
    leading space under `tokenizer`."""
    words = words if words is not None else CANDIDATE_WORDS
    kept_words, kept_ids = [], []
    for w in words:
        ids = tokenizer.encode(" " + w)
        if len(ids) == 1:
            kept_words.append(w)
            kept_ids.append(ids[0])
    return kept_words, kept_ids
