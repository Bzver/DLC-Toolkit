import numpy as np
from typing import Tuple, List

from .reviewer import Track_Correction_Dialog
from .mark_nav import get_prev_frame_in_list
from core.io import Frame_Extractor
from utils.pose import calculate_pose_centroids
from utils.helper import get_instance_count_per_frame, get_instances_on_current_frame
from utils.logger import logger
from utils.dataclass import Loaded_DLC_Data


DEBUG = False

class Track_Fixer:
    KALMAN_MAX_ERROR = 80.0
    AMBIGUITY_THRESHOLD = 10.0
    MAX_DIST_THRESHOLD = 120.0
    KALMAN_RESET_THRESHOLD = 3
    MIN_GAP = 10

    def __init__(
        self,
        pred_data_array: np.ndarray,
        persistent_idx: int,
        exit_zone: Tuple[int, int, int, int],
        dlc_data: Loaded_DLC_Data,
        extractor: Frame_Extractor,
        parent: object,
    ):
        if pred_data_array.shape[1] != 2:
            raise NotImplementedError("Track_Fixer supports exactly 2 instances.")

        self.pred_data_array = pred_data_array.copy()
        self.persistent_idx = persistent_idx
        self.exit_zone = exit_zone
        self.dlc_data = dlc_data
        self.extractor = extractor
        self.main = parent

        self.total_frames = self.pred_data_array.shape[0]
        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)
        self.inst_count_per_frame = get_instance_count_per_frame(self.pred_data_array)

        self.kalman_filters = [None, None]
        self.last_known_pos = np.full((2, 2), np.nan)
        self.kalman_failure_count = [0, 0]

        self.exit_windows = np.zeros((self.total_frames,), dtype=bool)
        self.exit_starts, self.exit_ends = [], []

        self.ambiguous_frames: List[int] = []

    def track_correction(self, start_idx: int = 0, end_idx: int = -1) -> np.ndarray:
        if end_idx == -1:
            end_idx = self.total_frames

        logger.debug(f"[TF] Starting track correction from frame {start_idx} to {end_idx}")
        
        self._detect_exit_windows(start_idx, end_idx)
        if self.exit_windows[0]:
            self.exit_starts = [0]
        self.exit_starts.extend(sorted(np.where((self.exit_windows[:-1] == False) & (self.exit_windows[1:] == True))[0] + 1))
        self.exit_ends = sorted(np.where((self.exit_windows[:-1] == True) & (self.exit_windows[1:] == False))[0] + 1)
        exit_starts_set = set(self.exit_starts)

        if self.exit_windows[self.total_frames-1]:
            self.exit_ends.append(self.total_frames-1)

        logger.debug(f"[TF] All exit window frames: \n {sorted(np.where(self.exit_windows)[0])}")

        self.ambiguous_frames = []
        for f in range(start_idx, end_idx):
            logger.debug(f"---------------- Frame {f} ----------------")
            self._enforce_exit_window_constraints(f)

            if not self.exit_windows[f]:
                success = self._correct_frame_with_hungarian(f)
                if not success:
                    self.ambiguous_frames.append(f)
                    logger.debug(f"[TF] Frame {f}: Ambiguous match, added to backtrack list")
            elif not f in exit_starts_set:
                curr_inst = get_instances_on_current_frame(self.pred_data_array, f)[0]
                self.last_known_pos[curr_inst] = self.centroids[f, curr_inst]
                self._update_kalman_with_observation(f, [curr_inst], [curr_inst])
                logger.debug(f"[TF] Frame {f}: In exit window, updated Kalman for present mouse")
            else:
                logger.debug(f"[TF] Frame {f}: Detected exit start, validating exited mouse")
                self._validate_exit_and_backtrack_if_needed(f)
                self.ambiguous_frames.clear() # This segment is over, clear existing ambiguous frames

        logger.debug("Track correction completed.")
        return self.pred_data_array

    def _detect_exit_windows(self, start_idx: int, end_idx: int):
        if DEBUG:
            exit_list = input("Paste in an existing exit window list, type 0 if don't skip:")
            exit_list = exit_list.replace('[', '').replace(']', '').replace('\n', ',')

            parts = []
            for segment in exit_list.split(','):
                subparts = segment.split()
                parts.extend(subparts)

            frame_nums = []
            for s in parts:
                s = s.strip()
                if s:
                    frame_nums.append(int(s))

            if len(frame_nums) != 1 or frame_nums[0] != 0:
                self.exit_windows[frame_nums] = True
                return

        logger.debug("[TF] Starting exit window detection")
        adjusted_count = self.inst_count_per_frame.copy()
        adjusted_count[adjusted_count == 0] = 1

        raw_segments = []
        i = start_idx
        while i < end_idx:
            val = adjusted_count[i]
            seg_start = i
            while i < end_idx and adjusted_count[i] == val:
                i += 1
            seg_end = i - 1
            raw_segments.append((seg_start, seg_end, val))

        i = 0
        while i < len(raw_segments):
            curr_start, curr_end, curr_val = raw_segments[i]
            if curr_val != 1:
                i += 1
                continue

            if i == 0 and curr_val == 1 and curr_end - curr_start >= self.MIN_GAP:
                logger.debug(
                    f"[TFEX] Auto-accepting initial solo segment ({curr_start}-{curr_end}) as pre-existing exit "
                    f"(duration={curr_end - curr_start + 1} > {self.MIN_GAP}). Assumes second mouse exited before recording."
                )
                exit_frame = curr_start
                valid_exit = True
            else:
                has_pre_2 = (i > 0 and raw_segments[i - 1][2] == 2)
                if not has_pre_2:
                    i += 1
                    continue

                if curr_end - curr_start < self.MIN_GAP:
                    ref_positions = self.centroids[curr_start-1]
                    min_cost = self._handle_two_observations(curr_end+1, ref_positions, get_min_cost=True)
                    if min_cost < self.MAX_DIST_THRESHOLD:
                        logger.debug(
                            f"[TFEX] Short exit segment ({curr_start}-{curr_end}, duration={curr_end - curr_start + 1}) "
                            f"rejected as noise: next frame {curr_end + 1} matches pre-exit state with low cost ({min_cost:.1f} < {self.MAX_DIST_THRESHOLD})"
                        )
                        i += 1
                        continue

                exit_frame = curr_start
                valid_exit = self._is_valid_exit_at(exit_frame)

            if not valid_exit:
                logger.debug(f"[TFEX] Exit at frame {exit_frame} failed validation, requesting user review")
                decision, confirmed_frame = self._launch_dialog(exit_frame, "exit")
                if not decision:
                    logger.debug(f"[TFEX] User rejected exit at frame {exit_frame}, skipping window")
                    i += 1
                    continue
                exit_frame = confirmed_frame
                curr_start = exit_frame
                logger.debug(f"[TFEX] User confirmed exit at frame {exit_frame}")

            window_accepted = False
            j = i + 1

            if j >= len(raw_segments):
                self.exit_windows[curr_start:curr_end+1] = True
                logger.debug(f"[TFEX] Accepted open-ended exit window: ({curr_start}, {curr_end})")
                window_accepted = True
            
            while j < len(raw_segments):
                if raw_segments[j][2] != 2:
                    j += 1
                    continue

                return_frame = raw_segments[j][0]

                if return_frame <= curr_start:
                    j += 1
                    continue

                during_frame = return_frame - 1
                exit_again_frame = raw_segments[j][1]

                if exit_again_frame - return_frame < self.MIN_GAP:
                    ref_positions = self.centroids[during_frame]
                    
                    actual_dr_frame = during_frame
                    while self.inst_count_per_frame[actual_dr_frame] == 0:
                        actual_dr_frame -= 1
                        ref_positions = self.centroids[actual_dr_frame]

                    actual_ext_frame = exit_again_frame + 1
                    while self.inst_count_per_frame[actual_ext_frame] == 0:
                        actual_ext_frame += 1

                    if self._handle_single_observation(actual_ext_frame, ref_positions, compare_only=True):
                        logger.debug(
                            f"[TFEX] Short re-entry segment ({return_frame}-{exit_again_frame}, duration={exit_again_frame - return_frame + 1}) "
                            f"rejected as spurious detection: frame {exit_again_frame + 1} consistent with post-return state"
                        )
                        j += 1
                        continue

                valid_return = self._is_valid_return_at(return_frame)
                if not valid_return:
                    logger.debug(f"[TFEX] Return candidate at frame {return_frame} failed validation, requesting user review")
                    decision, confirmed_frame = self._launch_dialog(return_frame, "return", curr_start)
                    if confirmed_frame - 1 < curr_start:
                        logger.debug(f"[TFEX] User denied the existence of an exit to begin with, aborting return search for this window.")
                        break
                    if decision:
                        self.exit_windows[curr_start:confirmed_frame] = True
                        logger.debug(f"[TFEX] Accepted exit window with user-confirmed return: ({curr_start}, {confirmed_frame - 1})")
                        window_accepted = True
                        break
                    else:
                        logger.debug(f"[TFEX] User rejected return at frame {return_frame}, searching for next return")
                else:
                    self.exit_windows[curr_start:return_frame] = True
                    logger.debug(f"[TFEX] Accepted exit window with auto-valid return: ({curr_start}, {return_frame - 1})")
                    window_accepted = True
                    break

                j += 1

            if not window_accepted:
                logger.debug(f"[TFEX] No valid return found for exit starting at {curr_start}, skipping window")

            i = j if window_accepted else i + 1

    def _is_valid_exit_at(self, exit_frame:int) -> bool:
        if (self.inst_count_per_frame[exit_frame-1] != 2 or self.inst_count_per_frame[exit_frame] not in (0, 1)):
            return False
        else:
            return np.any(self._in_zone_check(exit_frame-1))

    def _is_valid_return_at(self, return_frame:int) -> bool:
        if (self.inst_count_per_frame[return_frame-1] not in (0, 1) or self.inst_count_per_frame[return_frame] != 2):
            return False
        else:
            return np.any(self._in_zone_check(return_frame))

    def _enforce_exit_window_constraints(self, frame_idx: int):
        if not self.exit_windows[frame_idx]:
            return

        if frame_idx in self.exit_starts:
            exited_idx = None
            curr_valid = get_instances_on_current_frame(self.pred_data_array, frame_idx)

            if frame_idx != 0 and self.inst_count_per_frame[frame_idx-1] == 2:
                in_zone_status = self._in_zone_check(frame_idx-1)
                if np.sum(in_zone_status) == 1:
                    exited_idx = np.where(in_zone_status)[0][0]
                    logger.debug(f"[TFEX] Inferred exited mouse as ID {exited_idx} (only one in exit zone at frame {frame_idx-1})")

            if len(curr_valid) == 2:
                if exited_idx is None:
                    conf_all_kp = self.pred_data_array[frame_idx, :, 0::3]
                    conf_denan = np.nan_to_num(conf_all_kp, nan=0.0)
                    conf_inst = np.mean(conf_denan, axis=-1)
                    low_conf_inst = np.argmin(conf_inst)
                    logger.debug(f"[TFEX] Frame {frame_idx}: User-marked exit start but 2 instances detected; "
                                f"suppressing low-confidence instance {low_conf_inst} (conf: {conf_inst[low_conf_inst]:.2f})")
                    self._clear_instance_at_frame(frame_idx, low_conf_inst)
                else:
                    self._clear_instance_at_frame(frame_idx, exited_idx)
                    logger.debug(f"[TFEX] Cleared spurious detection of exited mouse {exited_idx}")
                self._update_during_exit_event(frame_idx)
            elif len(curr_valid) == 1:
                if exited_idx is None:
                    ref_pos = self._get_ref_pos(frame_idx)
                    self._handle_single_observation(frame_idx, ref_pos)
                elif curr_valid[0] == exited_idx:
                    self._swap_ids_in_frame(frame_idx)
                    logger.debug(f"[TFEX] Swapped IDs because exited mouse {exited_idx} was visible")
                    self._update_during_exit_event(frame_idx)
            return

        exited_idx = 1 - self.persistent_idx

        curr_valid = get_instances_on_current_frame(self.pred_data_array, frame_idx)

        if len(curr_valid) == 2:
            ref_pos = self.centroids[frame_idx-1, self.persistent_idx]
            dist_no_swap = np.linalg.norm(self.centroids[frame_idx, self.persistent_idx] - ref_pos)
            dist_swap = np.linalg.norm(self.centroids[frame_idx, exited_idx] - ref_pos)
            if dist_no_swap > dist_swap:
                self._swap_ids_in_frame(frame_idx)
            self._clear_instance_at_frame(frame_idx, exited_idx)
            logger.debug(f"[TFEX] Cleared spurious detection of exited mouse {exited_idx}")
        elif len(curr_valid) == 1:
            visible_idx = curr_valid[0]
            if visible_idx == exited_idx:
                self._swap_ids_in_frame(frame_idx)
                logger.debug(f"[TFEX] Swapped IDs because exited mouse {exited_idx} was visible")
        elif len(curr_valid) == 0:
            stayed_idx = 1 - exited_idx
            self.pred_data_array[frame_idx, stayed_idx] = self.pred_data_array[frame_idx - 1, stayed_idx]
            self.centroids[frame_idx, stayed_idx] = self.centroids[frame_idx - 1, stayed_idx]

        if self.kalman_filters[exited_idx] is not None:
            logger.debug(f"[TFEX] Terminating Kalman filter for Inst {exited_idx} (mouse exited)")
            self.kalman_filters[exited_idx] = None
            self.kalman_failure_count[exited_idx] = 0

        if not np.all(np.isnan(self.last_known_pos[exited_idx])):
            logger.debug(f"[TFEX] Terminating last known pos for Inst {exited_idx} (mouse exited)")
            self.last_known_pos[exited_idx, :] = np.nan

    def _in_zone_check(self, frame_idx:int) -> np.ndarray:
        in_zone_status = np.zeros((2,), dtype=bool)
        xmin, ymin, xmax, ymax = self.exit_zone
        for inst_idx in range(2):
            x, y = self.centroids[frame_idx, inst_idx]
            in_zone_status[inst_idx] = (xmin <= x <= xmax) and (ymin <= y <= ymax)
        return in_zone_status

    def _clear_instance_at_frame(self, frame_idx: int, inst_idx: int):
        self.pred_data_array[frame_idx, inst_idx, :] = np.nan
        self.centroids[frame_idx, inst_idx, :] = np.nan
        self.inst_count_per_frame[frame_idx] = len(get_instances_on_current_frame(self.pred_data_array, frame_idx))

    def _swap_ids_in_frame(self, frame_idx: int):
        self.pred_data_array[frame_idx] = self.pred_data_array[frame_idx, ::-1]
        self.centroids[frame_idx] = self.centroids[frame_idx, ::-1]

    def _update_during_exit_event(self, frame_idx:int):
        curr_inst = get_instances_on_current_frame(self.pred_data_array, frame_idx)[0]
        self.last_known_pos[curr_inst] = self.centroids[frame_idx, curr_inst]
        self._update_kalman_with_observation(frame_idx, [curr_inst], [curr_inst])

    def _correct_frame_with_hungarian(self, frame_idx: int) -> bool:
        n_obs = len(get_instances_on_current_frame(self.pred_data_array, frame_idx))

        if n_obs == 0:
            return True

        ref_pos = self._get_ref_pos(frame_idx)

        if n_obs == 1:
            result = self._handle_single_observation(frame_idx, ref_pos)
        elif n_obs == 2:
            result = self._handle_two_observations(frame_idx, ref_pos)
        
        frame_inst_after_update = get_instances_on_current_frame(self.pred_data_array, frame_idx)
        self.last_known_pos[frame_inst_after_update] = self.centroids[frame_idx, frame_inst_after_update]

        return result

    def _get_ref_pos(self, frame_idx:int, force_kalman:bool=False) -> np.ndarray:
        current_obs = self.centroids[frame_idx]

        kalman_refs = np.full((2, 2), np.nan)
        kalman_available = True
        for inst_idx in [0, 1]:
            if self.kalman_filters[inst_idx] is not None:
                state = self.kalman_filters[inst_idx].predict()
                kalman_refs[inst_idx] = state[:2]
            else:
                kalman_available = False

        logger.debug(f"[KALMAN] Kalman refs: Inst 0: ({kalman_refs[0,0]:.1f}, {kalman_refs[0,1]:.1f}), Inst 1: ({kalman_refs[1,0]:.1f}, {kalman_refs[1,1]:.1f})")

        if (kalman_available or force_kalman) and not np.isnan(kalman_refs).any():
            min_cost_kalman = self._compute_min_assignment_cost(kalman_refs, current_obs)
            if min_cost_kalman <= self.KALMAN_MAX_ERROR:
                logger.debug(f"[KALMAN] Kalman matching succeeded (min_cost={min_cost_kalman:.1f})")
                return kalman_refs

            logger.debug(f"[KALMAN] Kalman matching failed (min_cost={min_cost_kalman:.1f} > {self.KALMAN_MAX_ERROR}), falling back to last known positions")

        last0 = self.last_known_pos[0]
        last1 = self.last_known_pos[1]
        last0_str = f"({last0[0]:.1f}, {last0[1]:.1f})" if not np.isnan(last0).any() else "NaN"
        last1_str = f"({last1[0]:.1f}, {last1[1]:.1f})" if not np.isnan(last1).any() else "NaN"
        logger.debug(f"[KALMAN] Last known refs: Inst 0: {last0_str}, Inst 1: {last1_str}")

        return self.last_known_pos

    def _compute_min_assignment_cost(self, refs: np.ndarray, obs: np.ndarray) -> float:
        valid_obs = ~np.isnan(obs[:, 0])
        valid_obs_indices = np.where(valid_obs)[0]
        n_obs = np.sum(valid_obs)

        if n_obs == 1:
            obs_idx = valid_obs_indices[0]
            observation = obs[obs_idx]

            cost_0 = cost_1 = 1e6
            if not np.isnan(refs[0]).any():
                cost_0 = np.linalg.norm(refs[0] - observation)
            if not np.isnan(refs[1]).any():
                cost_1 = np.linalg.norm(refs[1] - observation)

            min_cost = min(cost_0, cost_1)

        elif n_obs == 2:
            cost_no_swap = cost_swap = 0.0
            cost_no_swap += np.linalg.norm(refs[0] - obs[0]) if not np.isnan(refs[0]).any() else 1e6
            cost_no_swap += np.linalg.norm(refs[1] - obs[1]) if not np.isnan(refs[1]).any() else 1e6
            cost_swap += np.linalg.norm(refs[0] - obs[1]) if not np.isnan(refs[0]).any() else 1e6
            cost_swap += np.linalg.norm(refs[1] - obs[0]) if not np.isnan(refs[1]).any() else 1e6
            min_cost = min(cost_no_swap, cost_swap)

        return min_cost

    def _handle_single_observation(self, frame_idx: int, ref_positions: np.ndarray, compare_only:bool=False) -> bool:
        obs_idx = get_instances_on_current_frame(self.pred_data_array, frame_idx)[0]
        obs_pos = self.centroids[frame_idx, obs_idx]

        costs = []
        candidates = []
        for candidate_idx in [0, 1]:
            if not np.isnan(ref_positions[candidate_idx, 0]):
                d = np.linalg.norm(obs_pos - ref_positions[candidate_idx])
                costs.append(d)
                candidates.append(candidate_idx)

        if not candidates:
            return False

        if len(candidates) == 1:
            if not compare_only:
                assigned_idx = candidates[0]
                if assigned_idx != obs_idx:
                    self._swap_ids_in_frame(frame_idx)
                    logger.debug(f"Frame {frame_idx}: Single obs assigned to {assigned_idx} (was {obs_idx}), swapped")
                self._update_kalman_with_observation(frame_idx, [assigned_idx], [assigned_idx])
            cost = costs[0]
            if cost <= self.MAX_DIST_THRESHOLD:
                return True
            else:
                logger.debug(f"Frame {frame_idx}: Single obs assignment cost too large: (cost: {cost:.1f} > {self.MAX_DIST_THRESHOLD})")
                return False
        else:
            c0, c1 = costs

            winner = candidates[0] if c0 < c1 else candidates[1]
            if winner != obs_idx:
                self._swap_ids_in_frame(frame_idx)
                logger.debug(f"Frame {frame_idx}: Single obs winner {winner} (was {obs_idx}), swapped")

            self._update_kalman_with_observation(frame_idx, [winner], [winner])

            if abs(c0 - c1) < self.AMBIGUITY_THRESHOLD:
                logger.debug(f"Frame {frame_idx}: Ambiguous single observation (costs {c0:.1f}, {c1:.1f})")
                return False

            return True

    def _handle_two_observations(self, frame_idx:int, ref_positions:np.ndarray, get_min_cost:bool=False) -> bool | float:
        valid_ref = np.where(np.all(~np.isnan(ref_positions), axis=-1))[0]
        n_ref = len(valid_ref)

        if n_ref == 1:
            ref_idx = valid_ref[0]
            cost_swap = np.linalg.norm(self.centroids[frame_idx, 1-ref_idx] - ref_positions[ref_idx])
            cost_no_swap = np.linalg.norm(self.centroids[frame_idx, ref_idx] - ref_positions[ref_idx])

        elif n_ref == 2:
            cost_matrix = np.full((2, 2), np.inf)
            for obs_idx in [0, 1]:
                for candidate_idx in [0, 1]:
                    d = np.linalg.norm(self.centroids[frame_idx, obs_idx] - ref_positions[candidate_idx])
                    cost_matrix[obs_idx, candidate_idx] = d

            cost_matrix_lap = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)

            cost_swap = cost_matrix_lap[0, 1] + cost_matrix_lap[1, 0]
            cost_no_swap = cost_matrix_lap[0, 0] + cost_matrix_lap[1, 1]

        else:
            return False

        if get_min_cost:
            return min(cost_swap, cost_no_swap)

        needs_swap = cost_swap < cost_no_swap
        if needs_swap:
            self._swap_ids_in_frame(frame_idx)
            logger.debug(f"Frame {frame_idx}: Two-obs swap needed (cost_swap={cost_swap:.1f} < no_swap={cost_no_swap:.1f})")

        self._update_kalman_with_observation(frame_idx, [0, 1], [0, 1])

        if abs(cost_swap - cost_no_swap) < self.AMBIGUITY_THRESHOLD:
            logger.debug(f"Frame {frame_idx}: Ambiguous two-obs match (Δcost={abs(cost_swap - cost_no_swap):.1f})")
            return False

        return True

    def _update_kalman_with_observation(self, frame_idx: int, observed_inst_ids: List[int], assigned_ids: List[int]):
        for obs_idx, global_idx in zip(observed_inst_ids, assigned_ids):
            z = self.centroids[frame_idx, obs_idx]
            if np.isnan(z).any():
                continue

            kf = self.kalman_filters[global_idx]

            if kf is None:
                self.kalman_filters[global_idx] = Kalman(initial_state=[z[0], z[1], 0.0, 0.0])
                self.kalman_failure_count[global_idx] = 0
            else:
                pred_state = kf.predict()
                pred_pos = pred_state[:2]
                error = np.linalg.norm(z - pred_pos)

                logger.debug(
                    f"[KALMAN] Global ID {global_idx} ← obs col {obs_idx} at ({z[0]:.1f}, {z[1]:.1f}), "
                    f"pred=({pred_pos[0]:.1f}, {pred_pos[1]:.1f}), error={error:.1f}"
                )

                if error > self.KALMAN_MAX_ERROR:
                    self.kalman_failure_count[global_idx] += 1
                    logger.debug(f"[KALMAN] Kalman error for ID {global_idx}: {error:.1f} > {self.KALMAN_MAX_ERROR}")
                    
                    if self.kalman_failure_count[global_idx] >= self.KALMAN_RESET_THRESHOLD:
                        logger.debug(f"[KALMAN] Resetting Kalman filter for ID {global_idx} after {self.kalman_failure_count[global_idx]} failures")
                        self.kalman_filters[global_idx] = Kalman(initial_state=[z[0], z[1], 0.0, 0.0])
                        self.kalman_failure_count[global_idx] = 0
                else:
                    kf.update(z)
                    self.kalman_failure_count[global_idx] = 0

    def _validate_exit_and_backtrack_if_needed(self, exit_start: int):
        if exit_start == 0:
            return

        curr_valid = ~np.isnan(self.centroids[exit_start, :, 0])

        if not np.sum(curr_valid) == 1:
            return

        vanished_idx = int(np.where(~curr_valid)[0][0])
        if vanished_idx == self.persistent_idx:
            logger.warning(f"Frame {exit_start}: Persistent mouse (ID {self.persistent_idx}) appears to have exited! Triggering backtracking.")
            self._backtrack_ambiguous_frames(exit_start)

    def _backtrack_ambiguous_frames(self, exit_start: int):
        candidates = sorted(set([f for f in self.ambiguous_frames if f < exit_start]))
        candidates.sort(reverse=True)

        if not candidates:
            logger.debug("Backtrack candidate list is empty, showing full segment.")
            decision, confirmed_frame = self._launch_dialog(
                frame_idx=exit_start-1,
                mode="swap",
                event_start_idx=get_prev_frame_in_list(self.exit_ends, exit_start-2)-10,
                event_end_idx=exit_start+11
                )

            if decision:
                logger.info(f"[BK] User confirmed swap at frame {confirmed_frame}, applying bulk swap")
                self._bulk_swap_from_frame(confirmed_frame)
                new_vanished = self._get_vanished_idx_at(exit_start)
                if new_vanished != self.persistent_idx:
                    logger.info(f"[BK] Backtracking successful: exited mouse is now ID {new_vanished} (not persistent)")
                    return
        else:
            logger.debug(f"[BK] Backtracking from exit start {exit_start} with candidate frames: {candidates}")

            for frame_idx in candidates:
                logger.debug(f"[BK] Reviewing ambiguous frame {frame_idx} for potential swap")
                decision, confirmed_frame = self._launch_dialog(
                    frame_idx=frame_idx,
                    mode="swap", 
                    event_start_idx=get_prev_frame_in_list(self.exit_ends, frame_idx)-10,
                    event_end_idx=exit_start+11)
                if decision:
                    logger.info(f"[BK] User confirmed swap at frame {confirmed_frame}, applying bulk swap")
                    self._bulk_swap_from_frame(confirmed_frame)
                    new_vanished = self._get_vanished_idx_at(exit_start)
                    if new_vanished != self.persistent_idx:
                        logger.info(f"[BK] Backtracking successful: exited mouse is now ID {new_vanished} (not persistent)")
                        return
                    else:
                        logger.warning(f"[BK] Backtracking at frame {confirmed_frame} did not resolve issue, trying earlier frame")

        logger.error(f"[BK] Backtracking failed for exit at {exit_start}: persistent mouse still appears to exit")

    def _get_vanished_idx_at(self, exit_start: int) -> int:
        curr_valid = ~np.isnan(self.centroids[exit_start, :, 0])
        if np.sum(curr_valid) == 1:
            return int(np.where(~curr_valid)[0][0])
        return -1
    
    def _bulk_swap_from_frame(self, start_frame: int):
        """Flip IDs for all frames from start_frame to end."""
        logger.debug(f"[BK] Bulk swapping IDs from frame {start_frame} to end")
        self.pred_data_array[start_frame:] = self.pred_data_array[start_frame:, ::-1]
        self.centroids[start_frame:] = self.centroids[start_frame:, ::-1]
        self.last_known_pos = self.last_known_pos[::-1]
        self.kalman_filters.reverse()
        self.kalman_failure_count.reverse()

    def _launch_dialog(self, frame_idx: int, mode: str, event_start_idx=None, event_end_idx=None) -> Tuple[bool, int]:
        dialog = Track_Correction_Dialog(
            dlc_data=self.dlc_data,
            extractor=self.extractor,
            pred_data_array=self.pred_data_array,
            current_frame_idx=frame_idx,
            mode=mode,
            event_start_idx=event_start_idx,
            event_end_idx=event_end_idx,
            parent=self.main
        )
        dialog.exec()

        decision, confirmed_frame = dialog.user_decision
        logger.debug(f"Dialog result for {mode} at frame {frame_idx}: decision={decision}, confirmed_frame={confirmed_frame}")
        return decision, confirmed_frame


class Kalman:
    def __init__(self, initial_state, dt: float = 1.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        q = 0.8
        dt2 = dt ** 2
        self.Q = np.array([[dt2*dt2/4, 0, dt2*dt/2, 0],
                           [0, dt2*dt2/4, 0, dt2*dt/2],
                           [dt2*dt/2, 0, dt2, 0],
                           [0, dt2*dt/2, 0, dt2]], dtype=float) * q
        self.R = np.eye(2) * 0.7
        self.x = np.array(initial_state).reshape(4, 1)
        self.P = np.eye(4) * 50

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.flatten()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P