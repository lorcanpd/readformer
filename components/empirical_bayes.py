import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
# ensure matplot lib can make plots on the server
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import gc  # Garbage Collector for memory management

matplotlib.use('Agg')


class EmpiricalBayes:
    """
    A class to perform Empirical Bayes analysis on model predictions,
    compute z-scores, local FDR, determine thresholds based on desired FDR,
    and generate corresponding visualizations and summary statistics.
    """

    def __init__(
            self, fold, phase_index, validation_output_dir, desired_fdr=0.01,
            sample_size=None, random_state=42, with_ground_truth=False
    ):
        """
        Initializes the EmpiricalBayes instance with necessary parameters.

        Args:
            fold (int): The current fold number for cross-validation.
            phase_index (int): The current validation phase index.
            validation_output_dir (str): Directory to save output files.
            desired_fdr (float): Desired false discovery rate (default: 0.01).
            sample_size (int, optional):
                Number of samples to randomly select for GMM fitting.
                If None, use all samples.
            random_state (int, optional): Seed for random number generator for reproducibility.
        """
        self.fold = fold
        self.phase_index = phase_index
        self.validation_output_dir = validation_output_dir
        self.desired_fdr = desired_fdr
        self.sample_size = sample_size
        self.random_state = random_state
        self.with_ground_truth = with_ground_truth

        # Construct the prediction file path
        self.prediction_filename = f"fold_{self.fold}_phase_{self.phase_index:03d}_predictions.csv"
        self.prediction_path = os.path.join(self.validation_output_dir, self.prediction_filename)

        # Construct output filenames
        self.plot_filename = f"fold_{self.fold}_phase_{self.phase_index:03d}_zscore_histogram.png"
        self.plot_path = os.path.join(self.validation_output_dir, self.plot_filename)

        self.results_filename = f"fold_{self.fold}_phase_{self.phase_index:03d}_results.csv"
        self.results_path = os.path.join(self.validation_output_dir, self.results_filename)

        # Setup logging
        os.makedirs(self.validation_output_dir, exist_ok=True)  # Ensure output directory exists
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(
                    self.validation_output_dir,
                    f"empirical_bayes_fold_{self.fold}_"
                    f"phase_{self.phase_index:03d}.log"
                ))
            ]
        )
        logging.info("Initialised EmpiricalBayes instance.")

    @staticmethod
    def compute_metrics(labels, preds):
        """
        Computes precision, recall, F1-score, and specificity.

        Args:
            labels (np.ndarray): True binary labels (0 or 1).
            preds (np.ndarray): Predicted binary labels (0 or 1).

        Returns:
            tuple: (precision, recall, f_score, specificity)
        """
        TP = np.sum((labels == 1) & (preds == 1))
        FP = np.sum((labels == 0) & (preds == 1))
        FN = np.sum((labels == 1) & (preds == 0))
        TN = np.sum((labels == 0) & (preds == 0))

        precision = TP / (TP + FP + 1e-9)
        recall = TP / (TP + FN + 1e-9)
        f_score = 2 * (precision * recall) / (precision + recall + 1e-9)
        specificity = TN / (TN + FP + 1e-9)

        return precision, recall, f_score, specificity

    def load_data(self):
        """
        Loads the prediction data from the CSV file.

        Returns:
            pd.DataFrame: The loaded prediction DataFrame.
        """
        if not os.path.isfile(self.prediction_path):
            logging.error(f"Prediction file not found at '{self.prediction_path}'.")
            raise FileNotFoundError(f"Prediction file not found at '{self.prediction_path}'.")

        df = pd.read_csv(self.prediction_path)
        logging.info(f"Loaded prediction data from '{self.prediction_path}'.")
        self.df = df  # Store DataFrame as a class attribute
        return df

    def compute_z_scores(self, df):
        """
        Computes posterior probabilities, log-odds, standard errors, and z-scores.

        Args:
            df (pd.DataFrame): The prediction DataFrame.

        Returns:
            tuple: (p, log_odds, std_error, z_scores)
        """
        alpha = df["alpha"].values
        beta = df["beta"].values

        p = alpha / (alpha + beta + 1e-12)
        log_odds = np.log(alpha / beta + 1e-12)  # Add epsilon to avoid log(0)
        std_error = ((alpha + beta) / np.sqrt((alpha + beta + 1) * alpha * beta + 1e-12))
        z_scores = log_odds / (std_error + 1e-12)  # Avoid division by zero

        logging.info("Computed p, log-odds, standard error, and z-scores.")
        return p, log_odds, std_error, z_scores

    def fit_gmm(self, z_scores):
        """
        Fits a two-component Gaussian Mixture Model to a randomly sampled subset of z-scores.

        Args:
            z_scores (np.ndarray): Array of z-scores.

        Returns:
            tuple: (gmm, null_component, alt_component)
        """
        if self.sample_size is not None:
            if self.sample_size > len(z_scores):
                logging.error(f"Sample size ({self.sample_size}) exceeds total number of samples ({len(z_scores)}).")
                raise ValueError(f"Sample size ({self.sample_size}) exceeds total number of samples ({len(z_scores)}).")

            np.random.seed(self.random_state)
            sampled_indices = np.random.choice(len(z_scores), size=self.sample_size, replace=False)
            sampled_z_scores = z_scores[sampled_indices]
            z_scores_reshaped = sampled_z_scores.reshape(-1, 1)
            logging.info(f"Randomly sampled {self.sample_size} z-scores for GMM fitting.")
        else:
            z_scores_reshaped = z_scores.reshape(-1, 1)
            logging.info(f"Using all {len(z_scores)} z-scores for GMM fitting.")

        # Fit GaussianMixture on the sampled data
        gmm = GaussianMixture(
            n_components=2,
            random_state=self.random_state,
            max_iter=100,
            covariance_type='full'
        ).fit(z_scores_reshaped)

        means = gmm.means_.flatten()
        covars = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        # Identify null component (most negative mean)
        null_component = np.argmin(means)
        alt_component = 1 - null_component

        logging.info(f"Fitted GaussianMixture: means={means}, covars={covars}, weights={weights}")
        logging.info(f"Identified null component: {null_component}, alt component: {alt_component}")

        # Clean up
        del z_scores_reshaped
        if self.sample_size:
            del sampled_indices
            del sampled_z_scores

        gc.collect()

        return gmm, null_component, alt_component

    def compute_lfdr(self, gmm, z_scores, null_component):
        """
        Computes the local false discovery rate (lfdr) for each z-score in batches.

        Args:
            gmm (GaussianMixture): The fitted GaussianMixture.
            z_scores (np.ndarray): Array of z-scores.
            null_component (int): Index of the null component.

        Returns:
            np.ndarray: Array of lfdr values.
        """
        z_scores_reshaped = z_scores.reshape(-1, 1)
        num_samples = len(z_scores)
        lfdr = np.empty(num_samples, dtype=np.float32)

        batch_size = 100000  # Adjust based on available memory
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_z = z_scores_reshaped[start:end]
            posterior_probs = gmm.predict_proba(batch_z)
            lfdr[start:end] = posterior_probs[:, null_component]
            logging.debug(f"Processed LFDR for samples {start} to {end}.")

            # Clean up
            del batch_z, posterior_probs
            gc.collect()

        logging.info("Computed local FDR for each z-score.")
        del z_scores_reshaped, gmm
        gc.collect()
        return lfdr

    def _determine_threshold_from_lfdr(self, z_scores, lfdr):
        """
        Determine the z-score threshold purely based on desired FDR,
        using local FDR (no ground truth needed).
        """
        sort_idx = np.argsort(z_scores)[::-1]
        sorted_lfdr = lfdr[sort_idx]
        cumulative_error_rate = np.cumsum(sorted_lfdr) / np.arange(1, len(sorted_lfdr) + 1)

        below_threshold = cumulative_error_rate <= self.desired_fdr
        if np.any(below_threshold):
            chosen_threshold_index = np.where(below_threshold)[0][-1]
            chosen_z_threshold = z_scores[sort_idx[chosen_threshold_index]]
            logging.info(
                f"Chosen z-threshold for FDR={self.desired_fdr}: "
                f"{chosen_z_threshold:.3f}")
            return chosen_z_threshold
        else:
            logging.warning(
                f"No threshold found that achieves FDR={self.desired_fdr}.")
            return None

    def _save_updated_csv(self, df, chosen_z_threshold):
        """
        Saves the updated CSV with z_score, lfdr, and called_mutation.
        """
        if "z_score" not in df.columns or "lfdr" not in df.columns:
            logging.error(
                "z_score and/or lfdr columns are missing from the DataFrame.")
            raise ValueError(
                "z_score and/or lfdr columns are missing from the DataFrame.")

        if chosen_z_threshold is not None:
            df["called_mutation"] = df["z_score"] > chosen_z_threshold
        else:
            df["called_mutation"] = False

        updated_filename = f"fold_{self.fold}_phase_{self.phase_index:03d}_epoch_{self.phase_index:03d}_predictions_with_zlfdr.csv"
        updated_path = os.path.join(self.validation_output_dir, updated_filename)
        df.to_csv(updated_path, index=False)
        logging.info(f"Saved updated predictions to '{updated_path}'.")

    def _save_results(self, chosen_z_threshold, precision, recall, f_score, specificity):
        """
        Saves threshold + metrics (if computed) to CSV. If with_ground_truth=False,
        metrics might be None or omitted.
        """
        results = {
            'fold': self.fold,
            'phase': self.phase_index,
            'desired_fdr': self.desired_fdr,
            'chosen_z_threshold': chosen_z_threshold,
            'precision': precision if precision is not None else "",
            'recall': recall if recall is not None else "",
            'f1_score': f_score if f_score is not None else "",
            'specificity': specificity if specificity is not None else ""
        }
        results_df = pd.DataFrame([results])
        results_df.to_csv(self.results_path, index=False)
        logging.info(f"Saved summary results to '{self.results_path}'.")

    def _plot_histogram(
            self, df, z_scores, gmm, null_component,
            chosen_z_threshold, precision, recall, f_score, specificity
    ):
        """
        Plots the z-score distribution. If with_ground_truth=True,
        we show a stacked histogram by label. Otherwise, just a single histogram of z-scores.
        """
        plt.figure(figsize=(10, 7))

        # Conditionally hue by label if ground truth is available
        if self.with_ground_truth and 'label' in df.columns:
            ax = sns.histplot(
                data=df,
                x='z_score',
                hue='label',
                bins=50,
                multiple='stack',
                stat='density',
                edgecolor='none',
                alpha=0.7
            )
        else:
            ax = sns.histplot(
                data=df,
                x='z_score',
                bins=50,
                stat='density',
                edgecolor='none',
                alpha=0.7
            )

        # Plot mixture components
        x = np.linspace(z_scores.min(), z_scores.max(), 1000)
        mixture_pdf = np.zeros_like(x)
        means = gmm.means_.flatten()
        covars = gmm.covariances_.flatten()
        weights = gmm.weights_.flatten()

        for i in range(2):
            pdf_i = (1 / np.sqrt(2 * np.pi * covars[i])) * np.exp(-(x - means[i]) ** 2 / (2 * covars[i]))
            component_pdf = weights[i] * pdf_i
            mixture_pdf += component_pdf
            comp_label = "Null" if i == null_component else "Alt"
            ax.plot(x, component_pdf, label=f"{comp_label} component", linestyle='--', zorder=5)

        ax.plot(x, mixture_pdf, 'k--', linewidth=2, label="Mixture", zorder=5)

        # Plot threshold & metrics annotation if threshold is found
        if chosen_z_threshold is not None:
            ax.axvline(
                x=chosen_z_threshold, color='red', linestyle='--', linewidth=2,
                label=f"Threshold (z={chosen_z_threshold:.2f})", zorder=10)

            # If with_ground_truth, we annotate metrics
            if self.with_ground_truth and precision is not None:
                annotation_text = (
                    f"FDR={self.desired_fdr:.4f}\n"
                    f"z={chosen_z_threshold:.4f}\n"
                    f"Prec={precision:.4f}, Recall={recall:.4f}, F1={f_score:.4f}\n"
                    f"Spec={specificity:.4f}"
                )
                ax.text(
                    chosen_z_threshold, ax.get_ylim()[1] * 0.9, annotation_text,
                    rotation=90, verticalalignment='top', horizontalalignment='center',
                    color='red', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
                )
            else:
                # If no ground truth, at least show the FDR & threshold
                annotation_text = f"FDR={self.desired_fdr:.3f}\nz={chosen_z_threshold:.3f}"
                ax.text(
                    chosen_z_threshold, ax.get_ylim()[1] * 0.9, annotation_text,
                    rotation=90, verticalalignment='top', horizontalalignment='center',
                    color='red', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='red')
                )

        ax.set_xlabel("Z-score")
        ax.set_ylabel("Density")
        title_str = "Z-score Distribution with Fitted Mixture"
        if self.with_ground_truth:
            title_str += " (Stacked by Label)"
        ax.set_title(title_str)
        handles, labels_legend = ax.get_legend_handles_labels()
        ax.legend(handles, labels_legend, loc='upper right')

        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
        logging.info(f"Saved histogram plot to '{self.plot_path}'.")

    def run(self):
        """
        Executes the Empirical Bayes analysis pipeline.
        """
        try:
            # Step 1: Load data
            df = self.load_data()

            # Step 2: Compute z-scores
            p, log_odds, std_error, z_scores = self.compute_z_scores(df)

            df['z_score'] = z_scores

            # Step 3: Fit GMM using GaussianMixture on a random sample
            gmm, null_component, alt_component = self.fit_gmm(z_scores)

            # Step 4: Compute lfdr
            lfdr = self.compute_lfdr(gmm, z_scores, null_component)

            df['lfdr'] = lfdr

            # Step 5: Determine threshold and compute metrics
            if self.with_ground_truth and 'label' not in df.columns:
                logging.error(
                    "Ground truth labels are required for threshold "
                    "determination."
                )
                raise ValueError(
                    "Ground truth labels are required for threshold "
                    "determination (add 'label' column)."
                )

            chosen_z_threshold, precision, recall, f_score, specificity = (None,) * 5
            chosen_z_threshold = self._determine_threshold_from_lfdr(z_scores, lfdr)

            if chosen_z_threshold is not None and self.with_ground_truth:
                labels = df["label"].values
                preds = (z_scores > chosen_z_threshold).astype(int)
                precision, recall, f_score, specificity = self.compute_metrics(labels, preds)

            # Step 6: Add 'called_mutation' column and save updated CSV
            self._save_updated_csv(df, chosen_z_threshold)

            # Step 7: Plot histogram
            self._plot_histogram(
                df, z_scores, gmm, null_component,
                chosen_z_threshold, precision, recall, f_score, specificity
            )

            if chosen_z_threshold is not None:
                # Step 8: Save summary results
                self._save_results(
                    chosen_z_threshold, precision, recall, f_score, specificity
                )
                if self.with_ground_truth:
                    logging.info(
                        f"Metrics at threshold {chosen_z_threshold:.4f} - "
                        f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
                        f"F1-Score: {f_score:.4f}, Specificity: {specificity:.4f}"
                    )

            logging.info("Empirical Bayes analysis completed successfully.")
        except Exception as e:
            logging.error(f"Error during Empirical Bayes analysis: {e}")
            raise e

