use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;

#[derive(Clone)]
pub struct TextDataset {
    pub sentences: Vec<String>,
}

impl Dataset<String> for TextDataset {
    fn get(&self, index: usize) -> Option<String> {
        self.sentences.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.sentences.len()
    }
}

#[derive(Clone, Debug)]
pub struct TextBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

// Batcher no longer owns a device in Burn 0.21 — it's passed into `batch()`
// so the same batcher can be reused across devices.
#[derive(Clone, Default)]
pub struct TextBatcher {}

impl<B: Backend> Batcher<B, String, TextBatch<B>> for TextBatcher {
    fn batch(&self, items: Vec<String>, device: &B::Device) -> TextBatch<B> {
        // Simple character-level tokenization mapping for illustration
        let mut tokens: Vec<Vec<i64>> = items
            .into_iter()
            .map(|s| s.chars().map(|c| (c as u32 % 256) as i64).collect())
            .collect();

        // Sentences have different lengths, but Tensor::stack requires every
        // item in the batch to share the same shape. Right-pad with 0 so all
        // sequences in the batch match the longest one.
        let max_len = tokens.iter().map(|t| t.len()).max().unwrap_or(0);
        for t in tokens.iter_mut() {
            t.resize(max_len, 0);
        }

        // Separate features (inputs) and target labels (offset by 1 token)
        let inputs_raw: Vec<Tensor<B, 1, Int>> = tokens
            .iter()
            .map(|t| Tensor::from_ints(&t[0..t.len() - 1][..], device))
            .collect();

        let targets_raw: Vec<Tensor<B, 1, Int>> = tokens
            .iter()
            .map(|t| Tensor::from_ints(&t[1..t.len()][..], device))
            .collect();

        TextBatch {
            inputs: Tensor::stack(inputs_raw, 0),
            targets: Tensor::stack(targets_raw, 0),
        }
    }
}