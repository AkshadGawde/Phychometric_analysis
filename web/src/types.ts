// Shapes of the events streamed by app/server.py (made in src/predict.py).

export type Modality = "visual" | "audio" | "text";
export type Trait = "openness" | "conscientiousness" | "extraversion" | "agreeableness" | "neuroticism";
export const TRAITS: Trait[] = ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"];
export const MODALITIES: Modality[] = ["visual", "audio", "text"];
export const TRAIT_LABEL: Record<Trait, string> = {
  openness: "Openness",
  conscientiousness: "Conscientiousness",
  extraversion: "Extraversion",
  agreeableness: "Agreeableness",
  neuroticism: "Emotional stability", // the dataset's "neuroticism" label is really stability
};
export const LANE_LABEL: Record<Modality, string> = { visual: "Face", audio: "Voice", text: "Words" };

export type StageName =
  | "ingest" | "audio_clean" | "audio_features" | "transcribe" | "visual" | "unimodal" | "fusion" | "explain";

export interface IngestData {
  has_audio: boolean; has_video: boolean; codecs: string[]; container: string;
  duration: number; sample_rate: number; raw_sec: number;
}

export interface AudioCleanData {
  raw_sec: number; clean_sec: number; trim_start_sec: number; trim_end_sec: number; removed_sec: number;
  gain: number; noise_floor_db: number; gated_share: number;
  raw_envelope: number[]; gated_envelope: number[]; clean_envelope: number[]; gate_curve: (number | null)[];
  settings: { noise_quiet_percent: number; gate_strength: number; trim_top_db: number; peak_level: number };
}

export interface VoiceMeasure { name: string; label: string; value: number; percentile: number }

export interface AudioFeaturesData {
  spectrogram: string; duration_sec: number;
  pitch: (number | null)[]; loudness_db: (number | null)[]; curve_sec: number; voiced_share: number;
  measures: VoiceMeasure[]; n_features: number; mfcc_means: number[];
  emotion_labels: string[]; emotion: number[];
  emotion_timeline: { start: number; end: number; probs: number[] }[];
  windows: { start: number; end: number }[];
}

export interface Token {
  w: string; start: number | null; end: number | null; p?: number; clean: string; removed: boolean;
  cues: { trait: Trait; side: "high" | "low" }[];
}

export interface TranscribeData {
  source: string; available: boolean; transcript: string; clean: string; n_words: number;
  tokens: Token[]; words_per_min: number;
  cue_words_found: Record<Trait, { high: string[]; low: string[] }>;
  emotion_labels: string[]; emotion: number[] | null; embedding_dims: number; note: string | null;
}

export interface FrameInfo {
  frame: string; crop: string; face: number[] | null; crop_box: number[];
  detected: boolean; replaced: boolean; emotion: number[];
}

export interface VisualData {
  available: boolean; reason?: string;
  frames?: FrameInfo[]; n_frames?: number; faces_found?: number; replaced?: number;
  content_box?: number[]; bars_removed?: boolean;
  emotion_labels?: string[]; emotion?: number[]; cnn_dims?: number;
}

export interface UnimodalData {
  available: Modality[]; skipped: Modality[];
  inputs: Partial<Record<Modality, number>>;
  classifiers: Partial<Record<Modality, Record<Trait, string>>>;
  prob: Partial<Record<Modality, Record<Trait, number>>>;
  score: Partial<Record<Modality, Record<Trait, number>>>;
}

export interface TestMetrics { acc: number; f1: number; auc: number }

export interface FusedTrait {
  label: string; subset: string; weights: Partial<Record<Modality, number>>;
  prob: number; percentile: number; spread: [number, number] | null; score: number; test: TestMetrics;
}

export interface FusionOption {
  weights: Partial<Record<Modality, number>>; quantiles: number[]; test: TestMetrics; val_auc: number;
}

export interface FusionData {
  available: Modality[]; choice_rule: string;
  unimodal_prob: Partial<Record<Modality, Record<Trait, number>>>;
  traits: Record<Trait, FusedTrait>;
  options: Record<string, Record<Trait, FusionOption>>;
  n_windows: number; held_out_clips: number;
}

export interface VoiceCue { feature: string; label: string; push: number; z: number; direction: "up" | "down"; percentile?: number }

export interface ExplainData {
  traits: Record<Trait, {
    label: string; modality_contribution: Partial<Record<Modality, number>>;
    voice_cues: VoiceCue[]; words: { high: string[]; low: string[] } | null;
  }>;
  note: string;
}

export interface StageDataMap {
  ingest: IngestData; audio_clean: AudioCleanData; audio_features: AudioFeaturesData;
  transcribe: TranscribeData; visual: VisualData; unimodal: UnimodalData; fusion: FusionData; explain: ExplainData;
}

export type ServerEvent =
  | { type: "queued"; position: number; ready: boolean; status: string }
  | { type: "stage_start"; stage: StageName; t: number }
  | { type: "stage"; stage: StageName; seconds: number; t: number; data: unknown }
  | { type: "error"; stage: StageName; message: string }
  | { type: "done"; seconds: number; timings: Record<string, number>; summary: unknown };

export interface Sample {
  name: string; title: string; description: string; audio: string | null; video: string | null; seconds: number;
}
