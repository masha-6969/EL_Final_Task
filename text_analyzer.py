#text_analyzer.py
import spacy
import os
from collections import defaultdict

# spaCyモデルのロード (初回のみダウンロードが必要)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    exit()

# --- (read_text_file, normalize_and_pos_tag は変更なし) ---
def read_text_file(filepath):
    """Reads text from the specified file path."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None

def normalize_and_pos_tag(text):
    """Normalizes text and returns a list of tokens with POS tags."""
    doc = nlp(text.lower()) # Process text in lowercase
    tokens_with_pos = []
    for token in doc:
        # Include only alphabetic or numeric tokens
        if token.is_alpha or token.is_digit:
            tokens_with_pos.append({
                'text': token.text,
                'pos': token.pos_ # Part-of-Speech tag
            })
    return tokens_with_pos


def find_common_patterns_improved(tokens1, tokens2, min_length=1):
    """
    Finds common patterns (sequences of words with matching POS tags)
    between two token lists.
    """
    patterns_with_pos = {} # Stores {pattern_text: pos_pattern_string} for each text

    def collect_patterns(tokens_list):
        sub_patterns = {}
        for i in range(len(tokens_list)):
            for j in range(i + min_length, len(tokens_list) + 1):
                sub_tokens = tokens_list[i:j]
                pattern_text = " ".join([t['text'] for t in sub_tokens])
                pos_pattern = "-".join([t['pos'] for t in sub_tokens])
                sub_patterns[pattern_text] = pos_pattern
        return sub_patterns

    sub_patterns1 = collect_patterns(tokens1)
    sub_patterns2 = collect_patterns(tokens2)

    common_patterns_data = {} # {pattern_text: {'pattern': ..., 'pos_pattern': ..., 'length': ...}}

    for pattern_text, pos_pattern1 in sub_patterns1.items():
        if pattern_text in sub_patterns2:
            pos_pattern2 = sub_patterns2[pattern_text]
            # Only include if both text and POS sequence match
            if pos_pattern1 == pos_pattern2:
                common_patterns_data[pattern_text] = {
                    'pattern': pattern_text,
                    'pos_pattern': pos_pattern1,
                    'length': len(pattern_text.split()),
                    # 'count' could be added here if counting occurrences in each text is desired
                }

    # Sort by length and filter out sub-patterns
    sorted_unique_patterns = sorted(common_patterns_data.values(), key=lambda x: x['length'], reverse=True)
    
    final_results = []
    for current_pattern_data in sorted_unique_patterns:
        is_sub_pattern_of_existing = False
        for existing_pattern_data in final_results:
            # Check if current pattern is a sub-string of an already added longer pattern
            if current_pattern_data['pattern'] in existing_pattern_data['pattern'] and \
               current_pattern_data['length'] < existing_pattern_data['length']:
                is_sub_pattern_of_existing = True
                break
        if not is_sub_pattern_of_existing:
            final_results.append(current_pattern_data)
    
    final_results.sort(key=lambda x: x['length'], reverse=True)
    
    return final_results


def find_pos_discrepancies_improved(tokens1, tokens2):
    """
    Finds words that exist in both token lists but have different POS tags.
    """
    word_to_pos_map1 = defaultdict(set)
    for token in tokens1:
        word_to_pos_map1[token['text']].add(token['pos'])

    word_to_pos_map2 = defaultdict(set)
    for token in tokens2:
        word_to_pos_map2[token['text']].add(token['pos'])

    discrepancies = set()

    for word in word_to_pos_map1.keys():
        if word in word_to_pos_map2:
            pos_set1 = word_to_pos_map1[word]
            pos_set2 = word_to_pos_map2[word]

            if pos_set1 != pos_set2:
                # Iterate through all combinations of POS tags that cause the discrepancy
                for p1 in pos_set1:
                    if p1 not in pos_set2:
                        for p2 in pos_set2:
                            discrepancies.add((word, p1, p2))
                for p2 in pos_set2:
                    if p2 not in pos_set1:
                        for p1 in pos_set1:
                             discrepancies.add((word, p1, p2))

    sorted_discrepancies = sorted(list(discrepancies), key=lambda x: (x[0], x[1], x[2]))
    formatted_discrepancies = [
        {'word': d[0], 'pos_text1': d[1], 'pos_text2': d[2]}
        for d in sorted_discrepancies
    ]
    return formatted_discrepancies

# --- 新規追加または大幅に修正する関数 ---
def analyze_phrase_patterns(text):
    """
    Given a text, analyze its phrase patterns using spaCy's dependency parser
    and noun chunks.
    Returns a list of identified phrase patterns and their types.
    """
    doc = nlp(text) # Use original casing for better phrase recognition if needed, or pass pre-normalized text.
                    # For this purpose, using the original casing from the 'pattern' field might be better.
                    # Let's assume 'text' here is a phrase like "the quick brown fox"

    phrases_info = []

    # 1. 名詞句 (Noun Chunks) の抽出
    # spaCyは自動的に名詞句を識別します。
    for chunk in doc.noun_chunks:
        phrases_info.append({
            'pattern': chunk.text,
            'type': 'Noun Phrase (NP)',
            'description': f"A noun phrase headed by '{chunk.root.text}' ('{chunk.root.pos_}') including its modifiers."
        })

    # 2. その他の句構造の推測（依存関係解析に基づく）
    # より複雑な句（動詞句、形容詞句、副詞句など）は、依存関係解析の結果から推測できます。
    # これは網羅的ではありませんが、一般的なパターンを捉えることができます。

    # 動詞句 (Verb Phrase - VP) の簡易的な識別
    # ルート動詞とその直接の依存関係にある要素を探す
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ == "ROOT": # 文の主要動詞
            # 動詞とその目的語、補語、副詞句などを結合して動詞句とする試み
            verb_phrase_tokens = [token.text]
            # 目的語 (dobj), 補語 (acomp, attr), 前置詞句 (prep) などを探す
            for child in token.children:
                if child.dep_ in ["dobj", "acomp", "attr", "prep", "advcl", "ccomp", "xcomp"]:
                    verb_phrase_tokens.append(child.text)
                    # 前置詞句などはさらにその子要素も含む
                    if child.dep_ == "prep":
                        for grand_child in child.children:
                            verb_phrase_tokens.append(grand_child.text)
            
            # ソートして元の順序に近づける (完全な順序保証は難しい)
            verb_phrase_text = " ".join(sorted(verb_phrase_tokens, key=lambda x: doc.text.find(x)))
            if verb_phrase_text != token.text: # 動詞単体でなければ
                phrases_info.append({
                    'pattern': verb_phrase_text,
                    'type': 'Verb Phrase (VP) - inferred',
                    'description': f"A verb phrase centered around '{token.text}' ('{token.pos_}') possibly including its objects/complements/adjuncts."
                })

        # 形容詞句 (Adjective Phrase - ADJP) の簡易的な識別
        # 形容詞をヘッドとする句
        if token.pos_ == "ADJ" and token.head.pos_ != "NOUN": # 名詞を直接修飾しない形容詞
            adj_phrase_tokens = [token.text]
            for child in token.children:
                if child.dep_ in ["advmod", "amod", "prep"]: # 副詞や前置詞句など
                    adj_phrase_tokens.append(child.text)
            
            adj_phrase_text = " ".join(sorted(adj_phrase_tokens, key=lambda x: doc.text.find(x)))
            if adj_phrase_text != token.text:
                phrases_info.append({
                    'pattern': adj_phrase_text,
                    'type': 'Adjective Phrase (ADJP) - inferred',
                    'description': f"An adjective phrase centered around '{token.text}' ('{token.pos_}')."
                })

        # 副詞句 (Adverb Phrase - ADVP) の簡易的な識別
        # 副詞をヘッドとする句
        if token.pos_ == "ADV" and token.head.pos_ != "VERB": # 動詞を直接修飾しない副詞
            adv_phrase_tokens = [token.text]
            for child in token.children:
                if child.dep_ in ["advmod", "prep"]: # 副詞や前置詞句など
                    adv_phrase_tokens.append(child.text)
            
            adv_phrase_text = " ".join(sorted(adv_phrase_tokens, key=lambda x: doc.text.find(x)))
            if adv_phrase_text != token.text:
                phrases_info.append({
                    'pattern': adv_phrase_text,
                    'type': 'Adverb Phrase (ADVP) - inferred',
                    'description': f"An adverb phrase centered around '{token.text}' ('{token.pos_}')."
                })
                
    return phrases_info


def write_results_to_file(filepath, common_patterns, pos_discrepancies, phrase_patterns_analysis):
    """Writes the analysis results to a file."""
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("---\n")
            f.write("## Shared Token Patterns (Text and POS Match)\n")
            f.write("---\n")
            if not common_patterns:
                f.write("No shared token patterns found.\n")
            else:
                for i, r in enumerate(common_patterns):
                    f.write(f"{i + 1}. \"{r['pattern']}\" (Length: {r['length']} tokens, POS: {r['pos_pattern']})\n")

            f.write("\n---\n")
            f.write("## POS Discrepancies (Same Word, Different POS)\n")
            f.write("---\n")
            if not pos_discrepancies:
                f.write("No POS discrepancies found.\n")
            else:
                f.write("Words with POS Discrepancies:\n")
                for d in pos_discrepancies:
                    f.write(f"  Word: \"{d['word']}\"\n")
                    f.write(f"    Text1 POS: {d['pos_text1']}\n")
                    f.write(f"    Text2 POS: {d['pos_text2']}\n")
                    f.write("\n")
            
            # --- 新しく追加する句形分析のセクション ---
            f.write("\n---\n")
            f.write("## Phrase Pattern Analysis of the Longest Common Pattern\n")
            f.write("---\n")
            if not phrase_patterns_analysis:
                f.write("No phrase patterns identified in the longest common pattern.\n")
            else:
                for pp in phrase_patterns_analysis:
                    f.write(f"  Pattern: \"{pp['pattern']}\"\n")
                    f.write(f"  Type: {pp['type']}\n")
                    f.write(f"  Description: {pp['description']}\n")
                    f.write("\n")

    except Exception as e:
        print(f"Error writing to file {filepath}: {e}")

# メイン処理
if __name__ == "__main__":
    text1_path = 'text1.txt'
    text2_path = 'text2.txt'
    output_path = 'out.txt'

    # --- 以下のテスト用のダミーテキストファイル生成部分はコメントアウトしました ---
    # with open(text1_path, 'w', encoding='utf-8') as f:
    #     f.write("The quick brown fox jumps over the lazy dog.\
    #     \n He runs quickly.\
    #     \n My project is a success.\
    #     \n I can run fast.\
    #     \n I like to run.\
    #     \n This is a big run.")
    # with open(text2_path, 'w', encoding='utf-8') as f:
    #     f.write("A brown fox quickly jumps over the lazy canine.\
    #             \n He runs.\
    #             \n Her project was a success.\
    #             \n He takes a morning run.\
    #             \n It is a very good run.")
    # --- コメントアウトここまで ---

    print(f"Reading {text1_path}...")
    text1_content = read_text_file(text1_path)
    print(f"Reading {text2_path}...")
    text2_content = read_text_file(text2_path)

    if text1_content is None or text2_content is None:
        print("Exiting due to file read errors. Please ensure 'text1.txt' and 'text2.txt' exist in the same directory.") # エラーメッセージを追記
    else:
        print("Normalizing and POS tagging text1...")
        tokens1 = normalize_and_pos_tag(text1_content)
        print("Normalizing and POS tagging text2...")
        tokens2 = normalize_and_pos_tag(text2_content)

        print("Finding common patterns...")
        common_patterns = find_common_patterns_improved(tokens1, tokens2)

        print("Finding POS discrepancies...")
        pos_discrepancies = find_pos_discrepancies_improved(tokens1, tokens2)

        # --- 句形分析の追加部分 ---
        phrase_patterns_analysis_results = []
        if common_patterns:
            # 最長の共通パターン（複数ある場合は最初のもの）を分析対象とする
            longest_common_pattern_text = common_patterns[0]['pattern']
            print(f"Analyzing phrase patterns for the longest common pattern: '{longest_common_pattern_text}'...")
            phrase_patterns_analysis_results = analyze_phrase_patterns(longest_common_pattern_text)
        else:
            print("No common patterns found to analyze for phrase patterns.")
        # --- ここまで ---

        print(f"Writing results to {output_path}...")
        write_results_to_file(output_path, common_patterns, pos_discrepancies, phrase_patterns_analysis_results)
        print("Done.")