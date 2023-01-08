from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys


def vectorize(Text):
    return TfidfVectorizer().fit_transform(Text).toarray()


def similarity(doc1, doc2):
    return cosine_similarity([doc1, doc2])


def check_plagiarism():
    for file in files:
        subfiles = file.strip().split(" ")
        notes = [open(_file, encoding='utf-8').read()
                 for _file in subfiles]
        vectors = vectorize(notes)
        s_vectors = list(zip(subfiles, vectors))
        for student_a, text_vector_a in s_vectors:
            new_vectors = s_vectors.copy()
            current_index = new_vectors.index((student_a, text_vector_a))
            del new_vectors[current_index]
            for student_b, text_vector_b in new_vectors:
                sim_score = similarity(text_vector_a, text_vector_b)[0][1]
                print(student_a, student_b, sim_score)
                with open(sys.argv[2], "a+", encoding='UTF-8') as file:
                    file.write(f"{sim_score}\n")
            break


if __name__ == '__main__':
    input_file = sys.argv[1]
    with open(input_file, 'r', encoding="UTF-8") as file:
        files = file.read().split("\n")
    check_plagiarism()
