import time

from model import RNet


def trainer(char_embedding_config, dataloader, pair_encoding_config, pointer_config, self_matching_config,
            sentence_encoding_config, start_time, word_embedding_config):
    model = RNet(char_embedding_config, word_embedding_config, sentence_encoding_config,
                 pair_encoding_config, self_matching_config, pointer_config)
    for batch in dataloader:
        question_ids, words, questions, contexts, answers, answers_texts = batch

        words.to_variable()
        questions.to_variable()
        contexts.to_variable()

        predict = model(words, questions, contexts)
        break
    print("finished in %f seconds" % (time.time() - start_time))
