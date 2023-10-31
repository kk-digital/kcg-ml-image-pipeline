




class Phrase:
    def __init__(self, id, phrase, clip_vector):
        self.id = id
        self.phrase = phrase
        self.clip_vector = clip_vector


class ClipServer:
    def __init__(self):
        self.id_counter = 0
        self.phrase_dictionary = {}

    def generate_id(self):
        new_id = self.id_counter
        self.id_counter = self.id_counter + 1

        return new_id

    def add_phrase(self, phrase):
        new_id = self.generate_id()
        clip_vector = compute_clip_vector(phrase)

        new_phrase = Phrase(new_id, phrase, clip_vector)
        self.phrase_dictionary[new_id] = new_phrase

        return new_phrase


    def get_phrase_list(self, offset, limit):
        result = []
        count = 0
        for key, value in self.phrase_dictionary.items():
            if count >= offset:
                if count < offset + limit:
                    result.append(value)
                else:
                    break
            count += 1
        return result