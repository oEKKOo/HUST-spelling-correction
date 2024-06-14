import re
import os
import math
from collections import Counter, defaultdict


# 读取词汇表
def load_vocab(file_path):
    with open(file_path, 'r') as f:
        vocab = set(f.read().split())
    return vocab

vocab = load_vocab('vocab.txt')

# 读取训练数据
def load_train_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data

train_data = load_train_data('text.txt')

# 生成拼写错误及其对应的正确单词对
def generate_error_pairs(data):
    error_pairs = []
    for line in data:
        parts = line.strip().split()
        if len(parts) > 3:
            error_word = parts[2]
            correct_word = parts[3]
            error_pairs.append((error_word, correct_word))
    return error_pairs

error_pairs = generate_error_pairs(train_data)

# 构建通道模型：基于拼写错误对
def build_channel_model(error_pairs):
    error_model = defaultdict(lambda: defaultdict(int))
    for error, correct in error_pairs:
        error_model[correct][error] += 1
    return error_model

channel_model = build_channel_model(error_pairs)

# 读取测试数据
def load_test_data(file_path):
    with open(file_path, 'r') as f:
        data = f.readlines()
    return data

test_data = load_test_data('testdata.txt')

# 生成候选纠正词
def generate_candidates(word, vocab):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    candidates = set(deletes + transposes + replaces + inserts)
    return candidates & vocab


def generate_candidates_fallback(word, vocab):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    candidates = set()
    for i in range(len(word)):
        # 删除
        candidate = word[:i] + word[i + 1:]
        candidates.add(candidate)

        # 替换
        for c in letters:
            candidate = word[:i] + c + word[i + 1:]
            candidates.add(candidate)

        # 插入
        for c in letters:
            candidate = word[:i] + c + word[i:]
            candidates.add(candidate)

        # 对调隔任意字符数的字母
        for j in range(i + 2, len(word)):
            candidate = word[:i] + word[j] + word[i + 1:j] + word[i] + word[j + 1:]
            candidates.add(candidate)

    return candidates & vocab



def generate_candidates_with_fallback(word, vocab):
    candidates = generate_candidates(word, vocab)
    if not candidates:
        candidates = generate_candidates_fallback(word, vocab)
    return candidates


# 训练n-gram语言模型
def build_ngram_model(corpus, n, vocab):
    def extract_ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    ngrams = Counter()
    for sentence in corpus:
        tokens = ['<s>'] * (n - 1) + [word for word in sentence.split() if word in vocab] + ['</s>']
        ngrams.update(extract_ngrams(tokens, n))
    total_ngrams = sum(ngrams.values())
    return ngrams, total_ngrams

# 读取语料库数据
def load_corpus(file_path):
    with open(file_path, 'r') as f:
        corpus = f.readlines()
    return corpus

corpus = load_corpus('text.txt')

# 构建3-gram语言模型
n = 3
ngram_model, total_ngrams = build_ngram_model(corpus, n, vocab)

# 计算句子的概率
def sentence_probability(sentence, ngram_model, total_ngrams, n):
    tokens = ['<s>'] * (n - 1) + sentence.split() + ['</s>']
    score = 0
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngram_count = ngram_model[ngram] + 1  # Laplace smoothing
        context_count = ngram_model[ngram[:-1]] + total_ngrams
        score += math.log(ngram_count / context_count)
    return score

# 处理缩写形式
def split_contraction(word):
    contractions = ["'ll", "'s", "n't", "'ve", "'re", "'d", "'m"]
    for contraction in contractions:
        if word.endswith(contraction):
            return word[:-len(contraction)], contraction
    return word, ''

# 拼写错误纠正
def correct_sentence(sentence, channel_model, ngram_model, total_ngrams, n, vocab):
    def replace_word_in_sentence(sentence, old_word, new_word):
        pattern = re.compile(r'\b' + re.escape(old_word) + r'\b')
        return pattern.sub(new_word, sentence)

    words = re.findall(r"(?:\d+\.\d+|\d{1,3}(?:,\d{3})*|\w+(?:'\w+)?|\S+|\s+)", sentence) # 保留分隔符（空格和标点）

    corrected_words = []
    for word in words:
        if re.match(r'\w+', word):  # 仅处理单词
            main_word, contraction = split_contraction(word)
            if main_word not in vocab:
                candidates = generate_candidates_with_fallback(main_word, vocab)
                if not candidates:
                    corrected_words.append(word)
                else:
                    best_candidate = max(candidates, key=lambda w: sentence_probability(replace_word_in_sentence(sentence, main_word, w), ngram_model, total_ngrams, n))
                    corrected_words.append(best_candidate + contraction)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ''.join(corrected_words)

# 处理测试数据并生成结果
results = []
for line in test_data:
    parts = line.strip().split('\t')
    if len(parts) == 3:
        id, num_errors, sentence = parts
        corrected_sentence = correct_sentence(sentence, channel_model, ngram_model, total_ngrams, n, vocab)
        results.append(f"{id}\t{corrected_sentence}")

# 保存结果
with open('result.txt', 'w') as f:
    for result in results:
        f.write(result + '\n')

# 评估
os.system('python eval.py')
