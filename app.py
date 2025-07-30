# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
import sys
import spacy # <-- spacy もここでインポート
from collections import defaultdict # <-- defaultdict もここでインポート

# ==== ここから、text_analyzer.py の全関数を正確に貼り付ける ====

# spaCyモデルのロード
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("SpaCy model 'en_core_web_sm' not found. Please run:")
    print("python -m spacy download en_core_web_sm")
    sys.exit("SpaCy model not loaded. Please install it.")

def read_text_file(filepath):
    # ... (text_analyzer.py の read_text_file の内容)
    pass # 実際の内容に置き換え

def normalize_and_pos_tag(text):
    # ... (text_analyzer.py の normalize_and_pos_tag の内容)
    pass # 実際の内容に置き換え

def find_common_patterns_improved(tokens1, tokens2, min_length=1):
    # ... (text_analyzer.py の find_common_patterns_improved の内容)
    pass # 実際の内容に置き換え

def find_pos_discrepancies_improved(tokens1, tokens2):
    # ... (text_analyzer.py の find_pos_discrepancies_improved の内容)
    pass # 実際の内容に置き換え

def analyze_phrase_patterns(text):
    # ... (text_analyzer.py の analyze_phrase_patterns の内容)
    pass # 実際の内容に置き換え

def write_results_to_file(filepath, common_patterns, pos_discrepancies, phrase_patterns_analysis):
    # ... (text_analyzer.py の write_results_to_file の内容)
    pass # 実際の内容に置き換え

# ==== ここまで、text_analyzer.py の全関数を正確に貼り付ける ====


app = Flask(__name__)

# ... (app.py の残りの部分)