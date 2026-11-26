# JAAD完全設定：Baseline対光フローなし

訓練コマンド: `python LOCAL_EXPERIMENT/scripts/flow_study/train_job.py LOCAL_EXPERIMENT/reports/flow_study/step4_jaad_noflow/jaad_corr_noflow_out15_s0_config.json`。評価: `python LOCAL_EXPERIMENT/scripts/flow_study/eval_job.py LOCAL_EXPERIMENT/reports/flow_study/step4_jaad_noflow/jaad_corr_noflow_out15_s0_config.json`。評価時間166.1秒。設定・所要時間はrun launch.json/complete.json、各epochはepochs.json。

{"val_samples": 616, "ADE": 11.61864797538751, "FDE": 22.979535518093154, "F1": 0.0, "Accuracy": 0.8831168831168831, "PR_AUC": 0.13876794131591796, "positive_ratio": 0.11688311688311688, "max_probability": 0.17778240144252777, "condition": "base", "commit_id": "fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0", "checkpoint": "LOCAL_EXPERIMENT/runs/jaad_corr_noflow_out15_s0/model_best0.001_512.pkl", "checkpoint_sha256": "173038bec1b4ee2a35c4f068c2d1ff9b1ddd6124ebcd6fd8223cf63584975af4", "seed": 0, "run": "jaad_corr_noflow_out15_s0", "training_commit_id": "fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0", "epochs": 22, "train_video_fraction": 1.0, "image_resize_width": 240, "image_resize_height": 426}

seedはモデル生成前reset、batch2、workers8、全val、eval/no_grad、平均ADE/FDE、標準F1/Accuracyとseries max P(crossing)のAPを報告。既存の作者F1式は使わない。

対応検定結果: {"N": 616, "mean_ADE_difference_noflow_minus_D1": -0.5168569805947217, "wilcoxon_p_two_sided": 0.0007240822666110799, "paired_bootstrap_CI95_low": -0.946276943257393, "paired_bootstrap_CI95_high": -0.08185214201931834, "video_cluster_bootstrap_CI95_low": -1.003364565665077, "video_cluster_bootstrap_CI95_high": 0.028191264085402377, "bootstrap_iterations": 10000, "seed": 0, "commit_id": "fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0"}。Wilcoxonは系列単位の両側検定。重複窓・同動画の相関があるため独立系列bootstrapに加えて動画単位cluster bootstrap CIを併記する。p>0.05だけでは同等性を証明しない。非劣性許容幅は指定されていないため、厳密な非劣性成立は主張しない。

Commit ID: `fa962404d7f18bee4b47084d0b8ffb4efcc8dfa0`。評価はvalidationのみ。testは読み込み・評価しない。
