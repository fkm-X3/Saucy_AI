mod data;
mod model;
mod training;

use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::DataLoaderBuilder;
use burn::optim::AdamConfig;
use burn::record::CompactRecorder;
use burn::train::metric::LossMetric;
use burn::train::renderer::CliMetricsRenderer;
use burn::train::{Learner, SupervisedTraining};

use crate::data::{TextBatcher, TextDataset};
use crate::model::LlmConfig;

use burn::train::EarlyStoppingStrategy;
use burn::train::metric::store::{Aggregate, EventStoreClient, Split};

// Names of every numeric metric registered on the learner (via
// `.metric_train_numeric(...)`) that we want to summarize each epoch.
const TRACKED_METRICS: &[&str] = &["Loss"];

#[derive(Clone, Default)]
pub struct EpochLogger {
    // Full metrics summary from the epoch that just finished, keyed by metric name.
    last_metrics: std::collections::HashMap<String, f64>,
}

impl EpochLogger {
    fn print_metrics(&self, header: &str) {
        if self.last_metrics.is_empty() {
            println!("{header} (no metrics recorded yet)\n");
            return;
        }
        println!("{header}");
        for name in TRACKED_METRICS {
            if let Some(value) = self.last_metrics.get(*name) {
                println!("    {name}: {:.6}", value);
            }
        }
        println!();
    }
}

impl EarlyStoppingStrategy for EpochLogger {
    fn should_stop(&mut self, epoch: usize, store: &EventStoreClient) -> bool {
        self.last_metrics.clear();
        for name in TRACKED_METRICS {
            if let Some(value) = store.find_metric(name, epoch, Aggregate::Mean, &Split::Train) {
                self.last_metrics.insert((*name).to_string(), value);
            }
        }

        self.print_metrics(&format!(
            "\nFinished Epoch {epoch} | Summary (moving on to the next epoch):"
        ));

        false // never actually stop training early
    }
}

fn main() {
    // Setup Backend Infrastructure (CPU only, via NdArray)
    type MyBackend = NdArray<f32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;
    let device = Default::default();

    // Initialize Hardware Configuration parameters
    let config = LlmConfig::new(256, 512, 8, 6, 128);
    let model = config.init::<MyAutodiffBackend>(&device);

    // Prepare Dummy Training Data
    let dataset_train = TextDataset {
        sentences: vec![
            "hello world from rust burn framework".to_string(),
            "building native deep learning applications".to_string(),
        ],
    };
    let dataset_valid = dataset_train.clone();

    let batcher = TextBatcher::default();

    let dataloader_train = DataLoaderBuilder::new(batcher.clone())
        .batch_size(2)
        .shuffle(42)
        .num_workers(1)
        .build(dataset_train);

    let dataloader_valid = DataLoaderBuilder::new(batcher)
        .batch_size(2)
        .shuffle(42)
        .num_workers(1)
        .build(dataset_valid);

    // Configure Optimization Parameters
    let optim = AdamConfig::new().init();

    // Configure and Launch Training Loop Execution
    let training = SupervisedTraining::new("./tmp/guide", dataloader_train, dataloader_valid)
        .metric_train_numeric(LossMetric::new())
        .renderer(CliMetricsRenderer::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(EpochLogger::default())
        .num_epochs(5);

    let learning_rate = 1e-4;
    let _result = training.launch(Learner::new(model, optim, learning_rate));
}