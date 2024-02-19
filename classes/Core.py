from mushroom_rl.core import Core
from collections import defaultdict


class CoreContinue(Core):
    """
        This class is used to store the state of the system at a specific check point.

    """
    def __init__(self, agent, mdp, callbacks_fit, callback_step, list_params):
        super().__init__(agent, mdp, callbacks_fit, callback_step)
        """
        Constructor.

        Args:
             step (int): the step number that should be saved;
             path (str): the directory to save the state of teh system.

        """
        self._state = mdp._state
        self._total_episodes_counter = list_params[0]
        self._total_steps_counter = list_params[1]
        self._current_episodes_counter = list_params[2]
        self._current_steps_counter = list_params[3]
        self._episode_steps = list_params[4]
        self._n_episodes = list_params[5]
        self._n_steps_per_fit = list_params[6]
        self._n_episodes_per_fit = list_params[7]

    def reset(self, initial_states=None):
        """
        Reset the state of the agent.

        """
        if initial_states is None \
                or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self.agent.episode_start()

        self._state = self._preprocess(self.mdp.reset(initial_state, self.mdp._true_action).copy())
        self.agent.next_action = None
        self._episode_steps = 0

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar,
                  episodes_progress_bar, render, initial_states):
        dataset = list()
        dataset_info = defaultdict(list)
        last = True
        while move_condition():
            if last:
                self.reset([self._state])

            sample, step_info = self._step(render)

            self.callback_step([sample])

            self._total_steps_counter += 1
            self._current_steps_counter += 1
            steps_progress_bar.update(1)

            if sample[-1]:
                self._total_episodes_counter += 1
                self._current_episodes_counter += 1
                episodes_progress_bar.update(1)

            dataset.append(sample)

            for key, value in step_info.items():
                dataset_info[key].append(value)

            if fit_condition():
                self.agent.fit(dataset, **dataset_info)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for c in self.callbacks_fit:
                    c(dataset)

                dataset = list()
                dataset_info = defaultdict(list)

            last = sample[-1]

        self.agent.stop()
        self.mdp.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return dataset, dataset_info


