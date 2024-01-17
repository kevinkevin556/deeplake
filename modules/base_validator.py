class BaseValidator:
    def __init__(self, metric, is_train=False, partially_labelled=False):
        self.metric = metric
        self.is_train = is_train
        self.partially_labelled = partially_labelled

        if is_train:
            self.pbar_description = "Validate ({global_step} Steps) (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"
        else:
            self.pbar_description = "Validate (Partially-labelled:{val_on_partial}) ({metric_name}={batch_metric:2.5f})"

    def __call__(self, module, dataloader, global_step: Optional[int] = None, **kwargs):
        return self.validation(module, dataloader, global_step, **kwargs)

    def validation(self, module, dataloader, global_step: Optional[int] = None, **kwargs):
        module.eval()
        val_metrics = {"ct": [], "mr": []}
        metric_means = {"mean": None, "ct": None, "mr": None}
        pbar = tqdm(dataloader, dynamic_ncols=True)

        with torch.no_grad():
            for batch in pbar:
                # Infer, decollate data into list of samples, and proprocess both predictions and labels
                images, masks = batch["image"].to(self.device), batch["label"].to(self.device)
                modality_label = batch["modality"][0]
                assert modality_label in ("ct", "mr"), f"Unknown/Invalid modality {modality_label}"

                infer_out = module.inference(images)
                samples = decollate_batch({"prediction": infer_out, "ground_truth": masks})

                # Post-processing
                if self.partially_labelled:
                    background_class = list(self.data_info["bg"][modality_label].keys())
                else:
                    background_class = None
                outputs, masks = get_output_and_mask(samples, self.num_classes, background_class)

                # Compute validation metrics
                self.metric(y_pred=outputs, y=masks)
                batch_metric = self.metric.aggregate().item()
                val_metrics[modality_label] += [batch_metric]
                self.metric.reset()

                # Update progressbar
                _info = {
                    "val_on_partial": self.partially_labelled,
                    "metric_name": self.metric.__class__.__name__,
                    "batch_metric": batch_metric,
                    "global_step": global_step,
                }
                _desc = self.pbar_description.format(**_info)
                pbar.set_description(_desc)

        metric_means["mean"] = np.mean(val_metrics["ct"] + val_metrics["mr"])
        metric_means["ct"] = np.mean(val_metrics["ct"]) if len(val_metrics["ct"]) > 0 else np.nan
        metric_means["mr"] = np.mean(val_metrics["mr"]) if len(val_metrics["mr"]) > 0 else np.nan
        return metric_means
