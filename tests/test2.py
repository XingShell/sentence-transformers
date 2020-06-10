from sentence_transformers.readers import NLIDataReader
from torch.utils.data import DataLoader
from sentence_transformers.models.tokenizer import tokenization_bert

from sentence_transformers.datasets import SentencesDataset
nli_reader = NLIDataReader('D:\\greedySchool\\myproject\\sentence-transformers\\sentence_transformers\\datasets\\AllNLI')

from tests.test1 import getmodel
model = getmodel()
InputExample = nli_reader.get_examples('dev.gz')
#         InputExample 的 []
#         self.guid = guid
#         self.texts = [text.strip() for text in texts]
#         self.label = label

train_data = SentencesDataset(InputExample, model)
print(InputExample[0].texts)
train_dataloader = DataLoader(train_data, shuffle=False, batch_size=1)
bertTokenizer = tokenization_bert.BertTokenizer("D:\\greedySchool\\myproject\\sentence-transformers\\sentence_transformers\\bert-base-uncased\\vocab.txt")

vocab = bertTokenizer.get_vocab()
vocab = dict(zip(vocab.values(), vocab.keys()))
print(vocab)

for x, y in train_dataloader:
    print(x)
    for l in x[0]:
        print(vocab[l.cpu().data.numpy()[0]], end='')

    exit()
# from sentence_transformers.losses import SoftmaxLoss
# train_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
#                                 num_labels=train_num_labels)