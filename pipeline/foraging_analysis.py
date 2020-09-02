import datajoint as dj
import numpy as np
import pandas as pd
import math
import warnings

from pipeline import get_schema_name, experiment

schema = dj.schema(get_schema_name('foraging_analysis'))

block_reward_ratio_increment_step = 10
block_reward_ratio_increment_window = 20
block_reward_ratio_increment_max = 200
bootstrapnum = 100
minimum_trial_per_block = 30

warnings.simplefilter(action='ignore', category=FutureWarning)


@schema
class SessionTaskProtocol(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_task_protocol : tinyint # the number of the dominant task protocol in the session
    session_real_foraging : bool # True if it is real foraging, false in case of pretraining
    """

    # Foraging sessions only
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task LIKE "foraging%"')

    def make(self, key):
        task_protocols = (experiment.BehaviorTrial & key).fetch('task_protocol')

        is_real_foraging = bool(experiment.SessionBlock.WaterPortRewardProbability & key
                                & 'reward_probability > 0 and reward_probability < 1')

        self.insert1({**key,
                      'session_task_protocol': np.median(task_protocols),
                      'session_real_foraging': is_real_foraging})


@schema
class TrialReactionTime(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    ---
    reaction_time=null : decimal(8,4) # reaction time in seconds (first lick relative to go cue) [-1 in case of ignore trials]
    """

    # Foraging sessions only
    key_source = experiment.BehaviorTrial & 'task LIKE "foraging%"'

    def make(self, key):
        gocue_time = (experiment.TrialEvent & key & 'trial_event_type = "go"').fetch1('trial_event_time')
        q_reaction_time = experiment.BehaviorTrial.aggr(experiment.ActionEvent & key
                                                        & 'action_event_type LIKE "%lick"'
                                                        & 'action_event_time > {}'.format(gocue_time),
                                                        reaction_time='min(action_event_time)')
        if q_reaction_time:
            key['reaction_time'] = q_reaction_time.fetch1('reaction_time')

        self.insert1(key)


# remove bias check trials from statistics # 03/25/20 NW added nobiascheck terms for hit, miss and ignore trial num
@schema
class SessionStats(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_total_trial_num = null          : int #number of trials
    session_block_num = null                : int #number of blocks, including bias check
    session_block_num_nobiascheck = null    : int # number of blocks, no bias check
    session_hit_num = null                  : int #number of hits
    session_hit_num_nobiascheck = null      : int # number of hits without bias check
    session_miss_num = null                 : int #number of misses
    session_miss_num_nobiascheck = null     : int # number of misses without bias check
    session_ignore_num = null               : int #number of ignores
    session_ignore_num_nobiascheck = null   : int #number of ignores without bias check
    session_ignore_trial_nums = null        : longblob #trial number of ignore trials
    session_autowater_num = null            : int #number of trials with autowaters
    session_length = null                   : decimal(10, 4) #length of the session in seconds
    session_bias_check_trial_num = null     : int #number of bias check trials
    session_bias_check_trial_idx = null     : longblob # index of bias check trials
    session_1st_3_ignores = null            : int #trialnum where the first three ignores happened in a row
    session_1st_2_ignores = null            : int #trialnum where the first three ignores happened in a row
    session_1st_ignore = null               : int #trialnum where the first ignore happened  
    session_biascheck_block = null          : longblob # the index of bias check blocks
    """

    # Foraging sessions only
    key_source = experiment.Session & (experiment.BehaviorTrial & 'task LIKE "foraging%"')

    def make(self, key):
        session_stats = {'session_total_trial_num': len(experiment.SessionTrial & key),
                         'session_block_num': len(experiment.SessionBlock & key),
                         'session_hit_num': len(experiment.BehaviorTrial & key & 'outcome = "hit"'),
                         'session_miss_num': len(experiment.BehaviorTrial & key & 'outcome = "miss"'),
                         'session_ignore_num': len(experiment.BehaviorTrial & key & 'outcome = "ignore"'),
                         'session_autowater_num': len(experiment.TrialNote & key & 'trial_note_type = "autowater"')}

        if session_stats['session_total_trial_num'] > 0:
            session_stats['session_length'] = float(((experiment.SessionTrial & key).fetch('stop_time')).max())
        else:
            session_stats['session_length'] = 0

        df_choices = pd.DataFrame({'outcome': (experiment.BehaviorTrial & key).fetch('outcome', order_by='trial')})

        if len(df_choices) > 0:
            realtraining = (experiment.SessionBlock.BlockTrial & key).aggr(
                experiment.SessionBlock.WaterPortRewardProbability, max_prob='max(reward_probability)').proj(
                ..., is_real_training='max_prob < 1').fetch('is_real_training', order_by='trial').astype(bool)

            realtraining2 = np.multiply(realtraining, 1)
            if not realtraining.any():
                session_stats['session_bias_check_trial_num'] = 0
                # print('all pretraining')
            else:
                session_stats['session_bias_check_trial_num'] = realtraining.argmax()
                session_stats['session_bias_check_trial_idx'] = np.array([x+1 for x, y in enumerate(realtraining2) if y == 0])
                #print(str(realtraining.values.argmax())+' out of '+str(keytoadd['session_trialnum']))

            if (df_choices['outcome'][session_stats['session_bias_check_trial_num']:] == 'ignore').values.any():
                session_stats['session_1st_ignore'] = (df_choices['outcome'][session_stats['session_bias_check_trial_num']:] == 'ignore').values.argmax()+session_stats['session_bias_check_trial_num']+1
                if (np.convolve([1,1],(df_choices['outcome'][session_stats['session_bias_check_trial_num']:] == 'ignore').values)==2).any():
                    session_stats['session_1st_2_ignores'] = (np.convolve([1,1],(df_choices['outcome'][session_stats['session_bias_check_trial_num']:] == 'ignore').values)==2).argmax() +session_stats['session_bias_check_trial_num']+1
                if (np.convolve([1,1,1],(df_choices['outcome'][session_stats['session_bias_check_trial_num']:] == 'ignore').values)==3).any():
                    session_stats['session_1st_3_ignores'] = (np.convolve([1,1,1],(df_choices['outcome'][session_stats['session_bias_check_trial_num']:] == 'ignore').values)==3).argmax() +session_stats['session_bias_check_trial_num']+1

            # get the hit, miss and ignore without bias check 03/25/20 NW
            session_stats['session_hit_num_nobiascheck'] = len(df_choices['outcome'][realtraining] == 'hit')
            session_stats['session_miss_num_nobiascheck'] = len(df_choices['outcome'][realtraining] == 'miss')
            session_stats['session_ignore_num_nobiascheck'] = len(df_choices['outcome'][realtraining] == 'ignore')

            # get the block num without bias check 03/25/20 NW
            if session_stats['session_block_num'] >= 1:
                blocks, is_bias_check_blocks = (experiment.SessionBlock & key).aggr(
                    experiment.SessionBlock.WaterPortRewardProbability & 'reward_probability = 1', c='count(*)').proj(
                    is_bias_check='c > 0').fetch('block', 'is_bias_check', order_by='block')
                is_bias_check_blocks = is_bias_check_blocks.astype(bool)

                session_stats['session_block_num_nobiascheck'] = sum(~is_bias_check_blocks)
                session_stats['session_biascheck_block'] = blocks[is_bias_check_blocks]
            else:
                session_stats['session_block_num_nobiascheck'] = 0

        self.insert1({**key, **session_stats})


@schema
class SessionMatchBias(dj.Computed):  # bias check removed,
    definition = """
    -> SessionStats
    """

    class WaterPortMatchBias(dj.Part):
        definition = """  # reward and choice fraction of this water-port w.r.t the sum of the other water-ports
        -> master
        -> experiment.WaterPort
        ---
        reward_fraction=null                : longblob      # lickport reward fraction from all blocks
        reward_fraction_first_tertile=null  : longblob      # from the first tertile blocks
        reward_fraction_second_tertile=null : longblob      # from the second tertile blocks
        reward_fraction_third_tertile=null  : longblob      # from the third tertile blocks
        choice_fraction=null                : longblob      # lickport choice fraction from all blocks        
        choice_fraction_first_tertile=null  : longblob      # from the first tertile blocks
        choice_fraction_second_tertile=null : longblob      # from the second tertile blocks
        choice_fraction_third_tertile=null  : longblob      # from the third tertile blocks
        match_idx=null                      : decimal(8,4)  # slope of log ratio from all blocks
        match_idx_first_tertile=null        : decimal(8,4)  # from the first tertile blocks
        match_idx_second_tertile=null       : decimal(8,4)  # from the second tertile blocks
        match_idx_third_tertile=null        : decimal(8,4)  # from the third tertile blocks
        bias=null                           : decimal(8,4)  # intercept of log ratio from all blocks
        bias_first_tertile=null             : decimal(8,4)  # from the first tertile blocks
        bias_second_tertile=null            : decimal(8,4)  # from the second tertile blocks
        bias_third_tertile=null             : decimal(8,4)  # from the third tertile blocks
        """

    def make(self, key):
        q_block_fraction = BlockFraction.WaterPortFraction & key

        fraction_attrs = ['reward_fraction', 'reward_fraction_first_tertile',
                          'reward_fraction_second_tertile', 'reward_fraction_third_tertile',
                          'choice_fraction', 'choice_fraction_first_tertile',
                          'choice_fraction_second_tertile', 'choice_fraction_third_tertile']
        session_bias = {}
        for water_port in experiment.WaterPort.fetch('water_port'):
            # ---- compute the reward and choice fraction, across blocks ----
            # query for this port
            this_port = (q_block_fraction & 'water_port = "{}"'.format(water_port))
            # query for other ports, take the sum of the reward and choice fraction of the other ports
            other_ports = BlockFraction.aggr(
                q_block_fraction & 'water_port != "{}"'.format(water_port),
                rw_sum='sum(block_reward_fraction)',
                rw1_sum='sum(block_reward_fraction_first_tertile)',
                rw2_sum='sum(block_reward_fraction_second_tertile)',
                rw3_sum='sum(block_reward_fraction_third_tertile)',
                choice_sum='sum(block_choice_fraction)',
                choice1_sum='sum(block_choice_fraction_first_tertile)',
                choice2_sum='sum(block_choice_fraction_second_tertile)',
                choice3_sum='sum(block_choice_fraction_third_tertile)')
            # merge and compute the fraction of this port over the sum of the other ports
            wp_block_bias = (this_port * other_ports).proj(
                reward_fraction='block_reward_fraction / rw_sum',
                reward_fraction_first_tertile='block_reward_fraction_first_tertile / rw1_sum',
                reward_fraction_second_tertile='block_reward_fraction_second_tertile / rw2_sum',
                reward_fraction_third_tertile='block_reward_fraction_third_tertile / rw3_sum',
                choice_fraction='block_choice_fraction / choice_sum',
                choice_fraction_first_tertile='block_choice_fraction_first_tertile / choice1_sum',
                choice_fraction_second_tertile='block_choice_fraction_second_tertile / choice2_sum',
                choice_fraction_third_tertile='block_choice_fraction_third_tertile / choice3_sum').fetch(
                *fraction_attrs, order_by='block')

            # taking the log2
            session_bias[water_port] = {attr: np.log2(attr_value.astype(float))
                                        for attr, attr_value in zip(fraction_attrs, wp_block_bias)}

            # ---- compute the match index and bias ----
            for tertile_suffix in ('', '_first_tertile', '_second_tertile', '_third_tertile'):
                reward_name = 'reward_fraction' + tertile_suffix
                choice_name = 'choice_fraction' + tertile_suffix
                # Ignore those with all NaNs or all Infs
                if (np.isfinite(session_bias[water_port][reward_name]).any()
                        and np.isfinite(session_bias[water_port][choice_name]).any()):
                    match_idx, bias = draw_bs_pairs_linreg(
                        session_bias[water_port][reward_name],
                        session_bias[water_port][choice_name], size=bootstrapnum)
                    session_bias[water_port]['match_idx' + tertile_suffix] = np.nanmean(match_idx)
                    session_bias[water_port]['bias' + tertile_suffix] = np.nanmean(bias)
                else:
                    session_bias[water_port].pop(reward_name)
                    session_bias[water_port].pop(choice_name)

        # ---- Insert ----
        self.insert1(key)
        # can't do list comprehension for batch insert because "session_bias" may have different fields
        for k, v in session_bias.items():
            self.WaterPortMatchBias.insert1({**key, 'water_port': k, **v})


@schema
class BlockStats(dj.Computed):
    definition = """
    -> experiment.SessionBlock
    ---
    block_trial_num : int # number of trials in block
    block_ignore_num : int # number of ignores
    block_reward_rate = null: decimal(8,4) # hits / (hits + misses)
    """

    def make(self, key):
        q_block_trials = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key

        block_stats = {'block_trial_num': len((experiment.SessionBlock.BlockTrial & key)),
                       'block_ignore_num': len(q_block_trials & 'outcome = "ignore"')}
        try:
            block_stats['block_reward_rate'] = len(q_block_trials & 'outcome = "hit"') / len(q_block_trials & 'outcome in ("hit", "miss")')
        except:
            pass
        self.insert1({**key, **block_stats})


@schema
class BlockFraction(dj.Computed):
    definition = """ # Block reward and choice fraction without bias check
    -> experiment.SessionBlock
    ---
    block_length: smallint #
    block_fraction: float
    first_tertile_fraction: float
    second_tertile_fraction: float
    third_tertile_fraction: float
    """

    class WaterPortFraction(dj.Part):
        definition = """
        -> master
        -> experiment.WaterPort
        ---
        block_reward_fraction=null                  : float     # lickport reward fraction from all trials
        block_reward_fraction_first_tertile=null    : float     # from the first tertile trials
        block_reward_fraction_second_tertile=null   : float     # from the second tertile blocks
        block_reward_fraction_third_tertile=null    : float     # from the third tertile blocks
        block_reward_fraction_incremental           : longblob  # from incremental trials
        block_choice_fraction=null                  : float
        block_choice_fraction_first_tertile=null    : float
        block_choice_fraction_second_tertile=null   : float
        block_choice_fraction_third_tertile=null    : float
        block_choice_fraction_incremental           : longblob        
        """

    _window_starts = (np.arange(block_reward_ratio_increment_window / 2,
                                block_reward_ratio_increment_max,
                                block_reward_ratio_increment_step, dtype=int)
                      - int(round(block_reward_ratio_increment_window / 2)))
    _window_ends = (np.arange(block_reward_ratio_increment_window / 2,
                              block_reward_ratio_increment_max,
                              block_reward_ratio_increment_step, dtype=int)
                    + int(round(block_reward_ratio_increment_window / 2)))

    @property
    def key_source(self):
        """
        Only process the blocks with:
         1. trial-count > minimum_trial_per_block
         2. is_real_training only
        """
        # trial-count > minimum_trial_per_block
        ks_tr_count = experiment.SessionBlock.aggr(experiment.SessionBlock.BlockTrial, tr_count='count(*)') & 'tr_count > {}'.format(minimum_trial_per_block)
        # is_real_training only
        ks_real_training = ks_tr_count - (experiment.SessionBlock.WaterPortRewardProbability & 'reward_probability >= 1')
        return ks_real_training

    def make(self, key):
        # To skip bias check trial 04/02/20 NW

        q_block_trial = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key

        block_rw = q_block_trial.proj(reward='outcome = "hit"').fetch('reward', order_by='trial')
        block_choice = (experiment.WaterPortChoice * q_block_trial).proj(
            non_null_wp='water_port IS NOT NULL').fetch('non_null_wp', order_by='trial')

        trialnum = len(block_rw)
        tertilelength = int(np.floor(trialnum / 3))

        # ---- whole-block fraction ----
        block_reward_fraction = dict(
            block_length=trialnum,
            block_fraction=block_rw.mean(),
            first_tertile_fraction=block_rw[:tertilelength].mean(),
            second_tertile_fraction=block_rw[tertilelength:2 * tertilelength].mean(),
            third_tertile_fraction=block_rw[-tertilelength:].mean())

        self.insert1({**key, **block_reward_fraction})

        # ---- water-port fraction ----
        wp_frac = {}
        for water_port in experiment.WaterPort.fetch('water_port'):
            # --- reward fraction ---
            rw = (experiment.WaterPortChoice * q_block_trial).proj(
                reward='water_port = "{}" AND outcome = "hit"'.format(water_port)).fetch('reward', order_by='trial').astype(float)
            wp_frac[water_port] = {
                'block_reward_fraction': rw.sum() / block_rw.sum() if block_rw.sum() else np.nan,
                'block_reward_fraction_first_tertile': (rw[:tertilelength].sum() / block_rw[:tertilelength].sum()
                                                        if block_rw[:tertilelength].sum() else np.nan),
                'block_reward_fraction_second_tertile': (rw[tertilelength:2 * tertilelength].sum() / block_rw[tertilelength:2 * tertilelength].sum()
                                                         if block_rw[tertilelength:2 * tertilelength].sum() else np.nan),
                'block_reward_fraction_third_tertile': (rw[-tertilelength:].sum() / block_rw[-tertilelength:].sum()
                                                        if block_rw[-tertilelength:].sum() else np.nan)}

            wp_frac[water_port]['block_reward_fraction_incremental'] = np.full(len(self._window_ends), np.nan)
            for i, (t_start, t_end) in enumerate(zip(self._window_starts, self._window_ends)):
                if trialnum >= t_end and block_rw[t_start:t_end].sum() > 0:
                    wp_frac[water_port]['block_reward_fraction_incremental'][i] = (rw[t_start:t_end].sum()
                                                                                   / block_rw[t_start:t_end].sum())
            # --- choice fraction ---
            choice = (experiment.WaterPortChoice * q_block_trial).proj(
                choice='water_port = "{}"'.format(water_port)).fetch('choice', order_by='trial').astype(float)

            wp_frac[water_port].update(**{
                'block_choice_fraction': choice.sum() / block_choice.sum() if block_choice.sum() else np.nan,
                'block_choice_fraction_first_tertile': (choice[:tertilelength].sum() / block_choice[:tertilelength].sum()
                                                        if block_choice[:tertilelength].sum() else np.nan),
                'block_choice_fraction_second_tertile': (choice[tertilelength:2 * tertilelength].sum() / block_choice[tertilelength:2 * tertilelength].sum()
                                                         if block_choice[tertilelength:2 * tertilelength].sum() else np.nan),
                'block_choice_fraction_third_tertile': (choice[-tertilelength:].sum() / block_choice[-tertilelength:].sum()
                                                        if block_choice[-tertilelength:].sum() else np.nan)})

            wp_frac[water_port]['block_choice_fraction_incremental'] = np.full(len(self._window_ends), np.nan)
            for i, (t_start, t_end) in enumerate(zip(self._window_starts, self._window_ends)):
                if trialnum >= t_end and block_choice[t_start:t_end].sum() > 0:
                    wp_frac[water_port]['block_choice_fraction_incremental'][i] = (choice[t_start:t_end].sum()
                                                                                   / block_choice[t_start:t_end].sum())

        self.WaterPortFraction.insert([{**key, 'water_port': wp, **wp_data} for wp, wp_data in wp_frac.items()])


@schema
class BlockEfficiency(dj.Computed):  # bias check excluded
    definition = """
    -> BlockFraction
    ---
    block_effi_one_p_reward: float                       # denominator = max of the reward assigned probability (no baiting)
    block_effi_one_p_reward_first_tertile: float         # first tertile
    block_effi_one_p_reward_second_tertile: float        # second tertile
    block_effi_one_p_reward_third_tertile: float         # third tertile
    block_effi_sum_p_reward: float                       # denominator = sum of the reward assigned probability (no baiting)
    block_effi_sum_p_reward_first_tertile: float         # first tertile
    block_effi_sum_p_reward_second_tertile: float        # second tertile
    block_effi_sum_p_reward_third_tertile: float         # third tertile
    block_effi_one_a_reward=null: float                  # denominator = max of the reward assigned probability + baiting)
    block_effi_one_a_reward_first_tertile=null: float    # first tertile
    block_effi_one_a_reward_second_tertile=null: float   # second tertile
    block_effi_one_a_reward_third_tertile=null: float    # third tertile
    block_effi_sum_a_reward=null: float                  # denominator = sum of the reward assigned probability + baiting)
    block_effi_sum_a_reward_first_tertile=null: float    # first tertile
    block_effi_sum_a_reward_second_tertile=null: float   # second tertile
    block_effi_sum_a_reward_third_tertile=null: float    # third tertile
    block_ideal_phat_greedy=null: float                  # denominator = Ideal-pHat-greedy
    regret_ideal_phat_greedy=null: float                 # Ideal-pHat-greedy - reward collected
    """

    def make(self, key):
        water_ports, rewards = (experiment.SessionBlock.WaterPortRewardProbability & key).fetch(
            'water_port', 'reward_probability')
        rewards = rewards.astype(float)
        max_prob_reward = np.nanmax(rewards)
        sum_prob_reward = np.nansum(rewards)

        q_block_trial = experiment.BehaviorTrial * experiment.SessionBlock.BlockTrial & key
        tertilelength = int(np.floor(len(q_block_trial) / 3))

        reward_available = pd.DataFrame({wp: (q_block_trial * experiment.TrialAvailableReward
                                              & {'water_port': wp}).fetch('reward_available', order_by='trial')
                                         for wp in water_ports})
        max_reward_available = reward_available.max(axis=1)
        max_reward_available_first = max_reward_available[:tertilelength]
        max_reward_available_second = max_reward_available[tertilelength:2 * tertilelength]
        max_reward_available_third = max_reward_available[-tertilelength:]

        sum_reward_available = reward_available.sum(axis=1)
        sum_reward_available_first = sum_reward_available[:tertilelength]
        sum_reward_available_second = sum_reward_available[tertilelength:2 * tertilelength]
        sum_reward_available_third = sum_reward_available[-tertilelength:]

        block_reward_fraction = (BlockFraction & key).fetch1()

        block_efficiency_data = dict(
            block_effi_one_p_reward=block_reward_fraction['block_fraction'] / max_prob_reward,
            block_effi_one_p_reward_first_tertile=block_reward_fraction['first_tertile_fraction'] / max_prob_reward,
            block_effi_one_p_reward_second_tertile=block_reward_fraction['second_tertile_fraction'] / max_prob_reward,
            block_effi_one_p_reward_third_tertile=block_reward_fraction['third_tertile_fraction'] / max_prob_reward,
            block_effi_sum_p_reward=block_reward_fraction['block_fraction'] / sum_prob_reward,
            block_effi_sum_p_reward_first_tertile=block_reward_fraction['first_tertile_fraction'] / sum_prob_reward,
            block_effi_sum_p_reward_second_tertile=block_reward_fraction['second_tertile_fraction'] / sum_prob_reward,
            block_effi_sum_p_reward_third_tertile=block_reward_fraction['third_tertile_fraction'] / sum_prob_reward,
            block_effi_one_a_reward=(block_reward_fraction['block_fraction'] / max_reward_available.mean()
                                     if max_reward_available.mean() else np.nan),
            block_effi_one_a_reward_first_tertile=(block_reward_fraction['first_tertile_fraction'] / max_reward_available_first.mean()
                                                   if max_reward_available_first.mean() else np.nan),
            block_effi_one_a_reward_second_tertile=(block_reward_fraction['second_tertile_fraction'] / max_reward_available_second.mean()
                                                    if max_reward_available_second.mean() else np.nan),
            block_effi_one_a_reward_third_tertile=(block_reward_fraction['third_tertile_fraction'] / max_reward_available_third.mean()
                                                   if max_reward_available_third.mean() else np.nan),
            block_effi_sum_a_reward=(block_reward_fraction['block_fraction'] / sum_reward_available.mean()
                                     if sum_reward_available.mean() else np.nan),
            block_effi_sum_a_reward_first_tertile=(block_reward_fraction['first_tertile_fraction'] / sum_reward_available_first.mean()
                                                   if sum_reward_available_first.mean() else np.nan),
            block_effi_sum_a_reward_second_tertile=(block_reward_fraction['second_tertile_fraction'] / sum_reward_available_second.mean()
                                                    if sum_reward_available_second.mean() else np.nan),
            block_effi_sum_a_reward_third_tertile=(block_reward_fraction['third_tertile_fraction'] / sum_reward_available_third.mean()
                                                   if sum_reward_available_third.mean() else np.nan))

        # Ideal-pHat-greedy  - only for blocks containing "left" and "right" port only
        if not len(np.setdiff1d(['right', 'left'], water_ports)):
            p1 = np.nanmax(rewards)
            p0 = np.nanmin(rewards)
            if p0 != 0:
                m_star_greedy = math.floor(math.log(1 - p1) / math.log(1 - p0))
                p_star_greedy = p1 + (1 - (1 - p0) ** (m_star_greedy + 1) - p1 ** 2) / (m_star_greedy + 1)

                block_efficiency_data.update(
                    block_ideal_phat_greedy=block_reward_fraction['block_fraction'] / p_star_greedy,
                    regret_ideal_phat_greedy=p_star_greedy - block_reward_fraction['block_fraction'])

        self.insert1({**key, **block_efficiency_data})


# ====================== HELPER FUNCTIONS ==========================

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""  # from serhan aya
    # Set up array of indices to sample from: inds
    inds = np.arange(len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = []
    bs_intercept_reps = []

    # Generate replicates
    for _ in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))  # sampling the indices (1d array requirement)
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        finite_idx = np.isfinite(bs_x) & np.isfinite(bs_y)  # consider only "finite" points: remove NaN and Inf points

        try:
            bs_slope, bs_intercept = np.polyfit(bs_x[finite_idx], bs_y[finite_idx], 1)
            bs_slope_reps.append(bs_slope)
            bs_intercept_reps.append(bs_intercept)
        except:
            pass

    return bs_slope_reps, bs_intercept_reps
