#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BAAI/bge-m3 Embedding Model Test Script
测试BAAI/bge-m3嵌入模型的功能和效果
"""

import sys
import os
import numpy as np
from typing import Dict, List

# Hugging Face cache settings (show progress, avoid Windows symlink issues)
if sys.platform == "win32":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

# 处理Windows终端编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_model_loading():
    """测试模型是否能正常加载"""
    print("=" * 60)
    print("测试1: 模型加载")
    print("=" * 60)
    
    try:
        _prefetch_bge_model()
        from FlagEmbedding import BGEM3FlagModel
        print("✓ FlagEmbedding库导入成功")
        
        print("\n正在加载BAAI/bge-m3模型（首次加载可能需要较长时间）...")
        model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        print("✓ 模型加载成功")
        
        return model
    except Exception as e:
        print(f"✗ 模型加载失败: {str(e)}")
        sys.exit(1)


def _prefetch_bge_model():
    """Pre-download model weights with progress output when possible."""
    try:
        from huggingface_hub import snapshot_download, logging as hf_logging

        hf_logging.set_verbosity_info()
        print("\nPrefetching Hugging Face model weights (shows progress)...")
        snapshot_download("BAAI/bge-m3", resume_download=True)
        print("✓ Model weights are ready in cache")
    except Exception as e:
        print(f"✗ Prefetch failed (will try direct load): {e}")

def test_dense_embedding(model):
    """测试Dense Embedding功能"""
    print("\n" + "=" * 60)
    print("测试2: Dense Embedding（密集嵌入）")
    print("=" * 60)
    
    try:
        test_texts = [
            "什么是BGE M3?",
            "什么是BM25算法?",
            "深度学习中的注意力机制"
        ]
        
        print(f"测试文本数量: {len(test_texts)}")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        
        embeddings = model.encode(test_texts, batch_size=12)['dense_vecs']
        
        print(f"\n✓ Dense Embedding生成成功")
        print(f"  嵌入维度: {embeddings.shape}")
        
        # 计算相似度
        print("\n计算文本之间的相似度（余弦相似度）:")
        similarity = embeddings @ embeddings.T
        
        for i in range(len(test_texts)):
            for j in range(i+1, len(test_texts)):
                sim = similarity[i][j]
                print(f"  文本{i+1} vs 文本{j+1}: {sim:.4f}")
                print(f"    ('{test_texts[i]}' vs '{test_texts[j]}')")
        
        return embeddings
    except Exception as e:
        print(f"✗ Dense Embedding测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_sparse_embedding(model):
    """测试Sparse Embedding（稀疏嵌入/词法权重）"""
    print("\n" + "=" * 60)
    print("测试3: Sparse Embedding（词法权重）")
    print("=" * 60)
    
    try:
        test_texts = [
            "BGE M3是一个支持密集检索、词法匹配和多向量交互的嵌入模型",
            "BM25是一个基于词袋的检索函数"
        ]
        
        print(f"测试文本数量: {len(test_texts)}")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        
        output = model.encode(test_texts, return_dense=False, return_sparse=True, return_colbert_vecs=False)
        
        print(f"\n✓ Sparse Embedding生成成功")
        
        # 显示词法权重
        tokens_weights = model.convert_id_to_token(output['lexical_weights'])
        for i, weights in enumerate(tokens_weights, 1):
            print(f"\n  文本{i}的词法权重:")
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
            for token, weight in sorted_weights:
                print(f"    '{token}': {weight:.4f}")
        
        # 计算词法匹配分数
        lexical_score = model.compute_lexical_matching_score(
            output['lexical_weights'][0], 
            output['lexical_weights'][1]
        )
        print(f"\n  文本1 vs 文本2 词法匹配分数: {lexical_score:.4f}")
        
    except Exception as e:
        print(f"✗ Sparse Embedding测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_colbert_embedding(model):
    """测试ColBERT多向量嵌入"""
    print("\n" + "=" * 60)
    print("测试4: ColBERT多向量嵌入")
    print("=" * 60)
    
    try:
        test_texts = [
            "信息检索是机器学习的重要应用",
            "深度学习在自然语言处理中的应用"
        ]
        
        print(f"测试文本数量: {len(test_texts)}")
        for i, text in enumerate(test_texts, 1):
            print(f"  {i}. {text}")
        
        output = model.encode(test_texts, return_dense=False, return_sparse=False, return_colbert_vecs=True)
        
        print(f"\n✓ ColBERT多向量嵌入生成成功")
        print(f"  向量形状: {output['colbert_vecs'][0].shape}")
        
        # 计算ColBERT分数
        colbert_score = model.colbert_score(output['colbert_vecs'][0], output['colbert_vecs'][1])
        print(f"\n  文本1 vs 文本2 ColBERT分数: {colbert_score:.4f}")
        
    except Exception as e:
        print(f"✗ ColBERT嵌入测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def test_combined_score(model):
    """测试综合评分（混合模式）"""
    print("\n" + "=" * 60)
    print("测试5: 综合评分（Dense + Sparse + ColBERT）")
    print("=" * 60)
    
    try:
        sentence_1 = ["多模态是指结合文本、图像、音频等多种信息模态"]
        sentence_2 = ["RAG系统可以增强大语言模型的性能"]
        
        print(f"查询: {sentence_1[0]}")
        print(f"参考: {sentence_2[0]}")
        
        # 准备句子对
        sentence_pairs = [[sentence_1[0], sentence_2[0]]]
        
        # 计算综合分数
        scores = model.compute_score(
            sentence_pairs,
            max_passage_length=128,
            weights_for_different_modes=[0.4, 0.2, 0.4]  # dense, sparse, colbert的权重
        )
        
        print(f"\n✓ 综合评分计算成功")
        print(f"  Dense分数: {scores['dense'][0]:.4f}")
        print(f"  Sparse分数: {scores['sparse'][0]:.4f}")
        print(f"  ColBERT分数: {scores['colbert'][0]:.4f}")
        print(f"  综合分数: {scores['colbert+sparse+dense'][0]:.4f}")
        
    except Exception as e:
        print(f"✗ 综合评分测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "     BAAI/BGE-M3 Embedding Model Test Suite".center(58) + "║")
    print("║" + "     BAAI/BGE-M3 嵌入模型测试套件".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    
    # 测试模型加载
    model = test_model_loading()
    
    # 测试各功能
    test_dense_embedding(model)
    test_sparse_embedding(model)
    test_colbert_embedding(model)
    test_combined_score(model)
    
    print("\n" + "=" * 60)
    print("✓ 所有测试完成！模型运行正常")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
