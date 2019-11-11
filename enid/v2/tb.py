class TBCallback(tf.keras.callbacks.TensorBoard):

    def __init__(self, evaluate_every, **kwargs):
        super(TBCallback, self).__init__()
        self.evaluate_every = evaluate_every

    def on_batch_end(self, batch, logs=None):
        """Writes scalar summaries for metrics on every training batch.
        Performs profiling if current batch is in profiler_batches.
        Arguments:
            batch: Integer, index of batch within the current epoch.
            logs: Dict. Metric results for this batch.
        """
        # Don't output batch_size and batch number as TensorBoard summaries
        logs = logs or {}
        self._samples_seen += logs.get('size', 1)
        samples_seen_since = self._samples_seen - self._samples_seen_at_last_write
        if samples_seen_since >= self.update_freq:
            self._log_metrics(logs, prefix='batch_', step=self._total_batches_seen)
            self._samples_seen_at_last_write = self._samples_seen
        self._total_batches_seen += 1
        if self._is_tracing:
            self._log_trace()
        elif (not self._is_tracing and
              self._total_batches_seen == self._profile_batch - 1):
            self._enable_trace()

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        step = epoch if self.update_freq == 'epoch' else self._samples_seen
        self._log_metrics(logs, prefix='epoch_', step=step)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

        if self.embeddings_freq and epoch % self.embeddings_freq == 0:
            self._log_embeddings(epoch)

    def _log_metrics(self, logs, prefix, step):
        """Writes metrics out as custom scalar summaries.
        Arguments:
                logs: Dict. Keys are scalar summary names, values are NumPy scalars.
                prefix: String. The prefix to apply to the scalar summary names.
                step: Int. The global step to use for TensorBoard.
        """
        if logs is None:
            logs = {}

        # Group metrics by the name of their associated file writer. Values
        # are lists of metrics, as (name, scalar_value) pairs.
        logs_by_writer = {
			self._train_run_name: [],
			self._validation_run_name: [],
        }
        validation_prefix = 'val_'
        for (name, value) in logs.items():
            if name in ('batch', 'size', 'num_steps'):
                # Scrub non-metric items.
                continue
            if name.startswith(validation_prefix):
                name = name[len(validation_prefix):]
                writer_name = self._validation_run_name
            else:
                writer_name = self._train_run_name
            name = prefix + name  # assign batch or epoch prefix
            logs_by_writer[writer_name].append((name, value))

        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                for writer_name in logs_by_writer:
                    these_logs = logs_by_writer[writer_name]
                    if not these_logs:
                        # Don't create a "validation" events file if we don't
                        # actually have any validation data.
                        continue
                    writer = self._get_writer(writer_name)
                    with writer.as_default():
                        for (name, value) in these_logs:
                            summary_ops_v2.scalar(name, value, step=step)
