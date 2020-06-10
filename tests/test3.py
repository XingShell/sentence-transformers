# from sentence_transformers import SentenceTransformer
# model = SentenceTransformer('bert-base-nli-mean-tokens')
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


from sentence_transformers.losses import SoftmaxLoss
train_loss = SoftmaxLoss(model=model, sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
                                num_labels=3)

InputExample = nli_reader.get_examples('dev.gz')
dev_data = SentencesDataset(InputExample, model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=40)
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
evaluator.device = "cpu"

model.fit(train_objectives=[(train_dataloader, train_loss)],
         evaluator=evaluator,
         epochs=1,
         evaluation_steps=5,
         warmup_steps=1000,
         output_path="./save"
         )