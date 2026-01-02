//! Embedding 缓存
//!
//! V-JEPA2 embedding 的环形缓冲区

use ndarray::{Array1, Array2};
use std::collections::VecDeque;
use std::sync::Arc;
use parking_lot::RwLock;

/// Embedding 缓存
pub struct EmbeddingCache {
    /// Embeddings 队列
    embeddings: Arc<RwLock<VecDeque<Array1<f32>>>>,
    /// 对应的时间戳
    timestamps: Arc<RwLock<VecDeque<f64>>>,
    /// 最大数量
    max_size: usize,
    /// Embedding 维度
    dim: usize,
}

impl EmbeddingCache {
    /// 创建新的 Embedding 缓存
    ///
    /// # Arguments
    /// * `max_size` - 最大缓存数量
    /// * `dim` - Embedding 维度 (例如: 1024 for V-JEPA2)
    pub fn new(max_size: usize, dim: usize) -> Self {
        Self {
            embeddings: Arc::new(RwLock::new(VecDeque::with_capacity(max_size))),
            timestamps: Arc::new(RwLock::new(VecDeque::with_capacity(max_size))),
            max_size,
            dim,
        }
    }

    /// 添加 embedding
    pub fn push(&self, embedding: Array1<f32>, timestamp: f64) {
        let mut embeddings = self.embeddings.write();
        let mut timestamps = self.timestamps.write();

        if embeddings.len() >= self.max_size {
            embeddings.pop_front();
            timestamps.pop_front();
        }

        embeddings.push_back(embedding);
        timestamps.push_back(timestamp);
    }

    /// 添加 embedding (从 slice)
    pub fn push_slice(&self, data: &[f32], timestamp: f64) {
        let embedding = Array1::from_vec(data.to_vec());
        self.push(embedding, timestamp);
    }

    /// 获取最近的 embeddings
    pub fn get_recent(&self, count: usize) -> (Vec<Array1<f32>>, Vec<f64>) {
        let embeddings = self.embeddings.read();
        let timestamps = self.timestamps.read();

        let n = count.min(embeddings.len());
        let embs: Vec<_> = embeddings.iter().rev().take(n).cloned().collect::<Vec<_>>().into_iter().rev().collect();
        let times: Vec<_> = timestamps.iter().rev().take(n).cloned().collect::<Vec<_>>().into_iter().rev().collect();

        (embs, times)
    }

    /// 获取所有 embeddings 作为矩阵
    pub fn get_all_as_matrix(&self) -> Option<Array2<f32>> {
        let embeddings = self.embeddings.read();
        if embeddings.is_empty() {
            return None;
        }

        let n = embeddings.len();
        let mut matrix = Array2::zeros((n, self.dim));

        for (i, emb) in embeddings.iter().enumerate() {
            matrix.row_mut(i).assign(emb);
        }

        Some(matrix)
    }

    /// 获取全局 embedding (平均池化)
    pub fn get_global_embedding(&self) -> Option<Array1<f32>> {
        let embeddings = self.embeddings.read();
        if embeddings.is_empty() {
            return None;
        }

        let mut sum = Array1::zeros(self.dim);
        for emb in embeddings.iter() {
            sum = sum + emb;
        }

        Some(sum / embeddings.len() as f32)
    }

    /// 清空缓存
    pub fn clear(&self) {
        self.embeddings.write().clear();
        self.timestamps.write().clear();
    }

    /// 获取数量
    pub fn len(&self) -> usize {
        self.embeddings.read().len()
    }

    /// 是否为空
    pub fn is_empty(&self) -> bool {
        self.embeddings.read().is_empty()
    }

    /// 计算两个 embedding 的余弦相似度
    pub fn cosine_similarity(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 0.0;
        }

        dot / (norm_a * norm_b)
    }

    /// 获取最后一个 embedding
    pub fn get_last(&self) -> Option<(Array1<f32>, f64)> {
        let embeddings = self.embeddings.read();
        let timestamps = self.timestamps.read();

        embeddings.back().cloned().zip(timestamps.back().cloned())
    }
}

impl Clone for EmbeddingCache {
    fn clone(&self) -> Self {
        Self {
            embeddings: Arc::clone(&self.embeddings),
            timestamps: Arc::clone(&self.timestamps),
            max_size: self.max_size,
            dim: self.dim,
        }
    }
}

impl Default for EmbeddingCache {
    fn default() -> Self {
        Self::new(60, 1024) // 60个 embedding, 1024维 (V-JEPA2 L)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_cache() {
        let cache = EmbeddingCache::new(5, 4);

        // 添加 embeddings
        for i in 0..10 {
            let emb = Array1::from_vec(vec![i as f32; 4]);
            cache.push(emb, i as f64 * 0.1);
        }

        // 检查大小
        assert_eq!(cache.len(), 5);

        // 获取全局 embedding
        let global = cache.get_global_embedding().unwrap();
        assert_eq!(global.len(), 4);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);

        assert!((EmbeddingCache::cosine_similarity(&a, &b) - 1.0).abs() < 0.01);
        assert!(EmbeddingCache::cosine_similarity(&a, &c).abs() < 0.01);
    }
}
