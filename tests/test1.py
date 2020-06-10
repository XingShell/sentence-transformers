# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bert-base-nli-mean-tokens')

from sentence_transformers.models import Transformer, Pooling
from sentence_transformers import SentenceTransformer

def getmodel():
    word_embedding_model = Transformer('D:\\greedySchool\\myproject\\sentence-transformers\\sentence_transformers\\bert-base-uncased'
                                       )
    pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model], device="cpu")
    return model

if __name__ == '__main__':
    sentences = ['This framework generates embeddings for each input sentence',
        'Sentences are passed as a list of string.',
        'The quick brown fox jumps over the lazy dog.']
    model = getmodel()
    sentence_embeddings = model.encode(sentences)
    # print(sentence_embeddings)

