import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook, SessionRunArgs
from tensorflow.python.training import training_util
from tensorflow.python.training.basic_session_run_hooks import SecondOrStepTimer


class ProfilerHook(SessionRunHook):
    def __init__ (self,
                  save_steps=None,
                  save_secs=None,
                  output_dir=""):
        self._output_tag = "step-{}"
        self._output_dir = output_dir
        self._timer = SecondOrStepTimer(
            every_secs=save_secs, every_steps=save_steps)

    def begin(self):
        self._next_step = None
        self._global_step_tensor = training_util.get_global_step()
        self._writer = tf.summary.FileWriter (self._output_dir, tf.get_default_graph())

        if self._global_step_tensor is None:
            raise RuntimeError("Global step should be created to use ProfilerHook.")

    def before_run(self, run_context):
        self._request_summary = (
            self._next_step is None or
            self._timer.should_trigger_for_step(self._next_step)
        )
        requests = {"global_step": self._global_step_tensor}
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            if self._request_summary else None)
        return SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._writer.add_run_metadata(
                run_values.run_metadata, self._output_tag.format(global_step))
            self._writer.flush()
        self._next_step = global_step + 1

    def end(self, session):
        self._writer.close()