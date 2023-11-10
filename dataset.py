import numpy as np
def get_dt():
    def batchify_data(data, batch_size=16, padding_token=-1):
        batches = []
        for idx in range(0, len(data), batch_size):
            # Create a batch with sequences
            batch = data[idx : idx + batch_size]

            max_batch_length = max(len(seq) for seq in batch)

            # Pad sequences to match the length of the longest sequence in the batch
            padded_batch = []
            for seq in batch:
                remaining_length = max_batch_length - len(seq)
                padded_seq = seq + [padding_token] * remaining_length
                padded_batch.append(padded_seq)

            batches.append(padded_batch)

        return batches
            
    from torchtext.data import get_tokenizer

    de_tokenizer = get_tokenizer('spacy', language='de')
    en_tokenizer = get_tokenizer('spacy', language='en')

    train_data = []

    content_train_x = open("train-x", "r")
    content_train_y = open("train-y", "r")


    for x, y in zip(content_train_x, content_train_y):
        en_tokens = en_tokenizer(x)
        de_tokens = de_tokenizer(y)
        train_data.append([en_tokens, de_tokens])

        print("English Tokens:", en_tokens)
        print("German Tokens:", de_tokens)

    train_dataloader = batchify_data(train_data)

    return train_dataloader, None  # No validation data