import os
import numpy as np
from sklearn.cluster import KMeans
from typing import Tuple, List, Literal, Dict, Optional

from .reviewer import Swap_Correction_Dialog, Exit_Reentry_Dialog
from core.io import Frame_Extractor
from ui import Progress_Indicator_Dialog
from utils.track import Kalman, swap_track
from utils.pose import calculate_pose_centroids, outlier_rotation, calculate_pose_array_rotations, calculate_pose_array_bbox
from utils.helper import get_instance_count_per_frame, get_instances_on_current_frame, get_prev_frame_in_list, indices_to_spans
from utils.dataclass import Loaded_DLC_Data, Blob_Config
from utils.logger import logger


class Track_Fixer:
    KALMAN_MAX_ERROR = 80.0
    AMBIGUITY_THRESHOLD = 20.0
    MAX_DIST_THRESHOLD = 120.0
    KALMAN_RESET_THRESHOLD = 3
    MIN_GAP = 5
    VANISH_CLUSTER_RADIUS = 200.0
    VANISH_CLUSTER_THRESHOLD = 10  
    MAX_REJECTION_HISTORY = 100
    USER_DIALOG_COOLDOWN = 20

    def __init__(
        self,
        pred_data_array: np.ndarray,
        persistent_idx: int,
        exit_zone: Tuple[int, int, int, int] | None,
        dlc_data: Loaded_DLC_Data,
        extractor: Frame_Extractor,
        avtomat: bool = False,
        parent = None,
    ):
        if pred_data_array.shape[1] != 2:
            raise NotImplementedError("Track_Fixer supports exactly 2 instances.")

        self.pred_data_array = pred_data_array.copy()
        self.persistent_idx = persistent_idx
        self.exit_zone = exit_zone
        self.dlc_data = dlc_data
        self.extractor = extractor
        self.avtomat = avtomat
        self.main = parent

        self.total_frames = self.pred_data_array.shape[0]
        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)
        self.inst_count_per_frame = get_instance_count_per_frame(self.pred_data_array)

        self.kalman_filters = [None, None]
        self.last_known_pos = np.full((2, 2), np.nan)
        self.kalman_failure_count = [0, 0]

        self.exit_windows = np.zeros((self.total_frames,), dtype=bool)
        self.exit_starts, self.exit_ends = [], []

        self.rejected_vanish_positions = []
        self.last_exit_frame_shown = self.last_return_frame_shown = -self.USER_DIALOG_COOLDOWN

    def track_correction(self, start_idx: int = 0, end_idx: int = -1) -> np.ndarray:
        if end_idx == -1:
            end_idx = self.total_frames

        logger.info(f"[TF] Starting track correction from frame {start_idx} to {end_idx}")

        x1, y1, x2, y2 = self.exit_zone
        logger.debug(f"Exit zone: ({x1},{y1})-({x2},{y2}), persistent_idx: {self.persistent_idx}")
        
        self._detect_exit_windows(start_idx, end_idx)
        if self.exit_windows[0]:
            self.exit_starts = [0]
        self.exit_starts.extend(sorted(np.where((self.exit_windows[:-1] == False) & (self.exit_windows[1:] == True))[0] + 1))
        self.exit_ends = sorted(np.where((self.exit_windows[:-1] == True) & (self.exit_windows[1:] == False))[0] + 1)
        exit_starts_set = set(self.exit_starts)

        if self.exit_windows[self.total_frames-1]:
            self.exit_ends.append(self.total_frames-1)

        logger.debug(f"[TF] All exit window frames:   {sorted(np.where(self.exit_windows)[0])}")

        ambiguous_frames = []
        for f in range(start_idx, end_idx):
            logger.debug(f"---------------- Frame {f} ----------------")
            self._enforce_exit_window_constraints(f)

            logger.debug(f"[TF] Pre-correct pos: Inst 0: ({self.centroids[f,0,0]:.1f}, {self.centroids[f,0,1]:.1f}), Inst 1: ({self.centroids[f,1,0]:.1f}, {self.centroids[f,1,1]:.1f})")
            if not self.exit_windows[f]:
                success = self._correct_frame_with_hungarian(f)
                if not success:
                    ambiguous_frames.append(f)
                    logger.debug(f"[TF] Ambiguous match, added to the list")
            elif not f in exit_starts_set:
                curr_inst_list = get_instances_on_current_frame(self.pred_data_array, f)
                if curr_inst_list:
                    curr_inst = curr_inst_list[0]
                    self.last_known_pos[curr_inst] = self.centroids[f, curr_inst]
                    self._update_kalman_with_observation(f, [curr_inst], [curr_inst])
                    logger.debug("[TF] In exit window, updated Kalman for present mouse")
            else:
                logger.debug("[TF] Detected exit start, validating exited mouse")

                if self._validate_at_exit(f):
                    logger.warning(f"Frame {f}: Persistent mouse (ID {self.persistent_idx}) appears to have exited! Triggering reviewing.")
                    self._review_ambiguous_frames(ambiguous_frames, f)
                ambiguous_frames.clear()

        logger.debug("Track correction completed.")
        return self.pred_data_array

    def _detect_exit_windows(self, start_idx: int, end_idx: int):
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
        last_return_idx = None

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
                if self._auto_reject_check(exit_frame):
                    logger.debug(f"[TFEX] Auto-rejecting exit at frame {exit_frame} due to false positive history.")
                    i += 1
                    continue

                logger.debug(f"[TFEX] Exit at frame {exit_frame} failed validation, requesting user review")

                decision, confirmed_frame = self._launch_dialog_exit_return(
                    frame_idx=exit_frame,
                    mode="exit",
                    event_start_idx=last_return_idx if last_return_idx else exit_frame-20,
                    )
                if not decision:
                    logger.debug(f"[TFEX] User rejected exit at frame {exit_frame}, skipping window")
                    self._log_vanishing_pos(exit_frame)
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
                avg_instances = np.sum(self.inst_count_per_frame[exit_frame:return_frame])/(return_frame-exit_frame) 
                risky_window = avg_instances > 1.5
                logger.debug(f"[TFEX] Exit window {exit_frame}-{return_frame}: avg instances = {avg_instances:.2f}, risky = {risky_window}")

                if not valid_return or risky_window:
                    logger.debug(f"[TFEX] Return candidate at frame {return_frame} failed validation, requesting user review")
                    decision, confirmed_frame = self._launch_dialog_exit_return(return_frame, "return", curr_start)
                    if confirmed_frame - 1 < curr_start:
                        logger.debug(f"[TFEX] User denied the existence of an exit to begin with, aborting return search for this window.")
                        break
                    if decision:
                        self.exit_windows[curr_start:confirmed_frame] = True
                        last_return_idx = confirmed_frame
                        logger.debug(f"[TFEX] Accepted exit window with user-confirmed return: ({curr_start}, {confirmed_frame - 1})")
                        window_accepted = True
                        break
                    else:
                        logger.debug(f"[TFEX] User rejected return at frame {return_frame}, searching for next return")
                else:
                    last_return_idx = return_frame
                    self.exit_windows[curr_start:return_frame] = True
                    logger.debug(f"[TFEX] Accepted exit window with auto-valid return: ({curr_start}, {return_frame - 1})")
                    window_accepted = True
                    break

                j += 1

            if not window_accepted:
                logger.debug(f"[TFEX] No valid return found for exit starting at {curr_start}, skipping window")

            i = j if window_accepted else i + 1

    def _is_valid_exit_at(self, exit_frame: int) -> bool:
        if (self.inst_count_per_frame[exit_frame-1] != 2 or self.inst_count_per_frame[exit_frame] not in (0, 1)):
            logger.debug(f"[EXIT] Frame {exit_frame}: Invalid exit conditions (pre={self.inst_count_per_frame[exit_frame-1]}, curr={self.inst_count_per_frame[exit_frame]})")
            return False
        
        vanishing_pos = self._find_vanishing_pos(exit_frame)
        if vanishing_pos is None:
            logger.debug(f"[EXIT] Frame {exit_frame}: Could not determine vanishing position")
            return False

        valid = self._in_zone_check_pos(vanishing_pos)
        logger.debug(f"[EXIT] Frame {exit_frame}: Exit validation {'PASSED' if valid else 'FAILED'} - vanishing pos ({vanishing_pos[0]:.1f}, {vanishing_pos[1]:.1f})")
        return valid

    def _is_valid_return_at(self, return_frame: int) -> bool:
        if (self.inst_count_per_frame[return_frame-1] not in (0, 1) or self.inst_count_per_frame[return_frame] != 2):
            logger.debug(f"[RETURN] Frame {return_frame}: Invalid return conditions (pre={self.inst_count_per_frame[return_frame-1]}, curr={self.inst_count_per_frame[return_frame]})")
            return False

        returning_pos = self._find_returning_pos(return_frame)
        if returning_pos is None:
            logger.debug(f"[RETURN] Frame {return_frame}: Could not determine returning position")
            return False

        valid = self._in_zone_check_pos(returning_pos)
        logger.debug(f"[RETURN] Frame {return_frame}: Return validation {'PASSED' if valid else 'FAILED'} - returning pos ({returning_pos[0]:.1f}, {returning_pos[1]:.1f})")
        return valid

    def _log_vanishing_pos(self, exit_frame:int):
        if exit_frame == 0 or self.inst_count_per_frame[exit_frame - 1] != 2:
            return
        
        vanishing_pos = self._find_vanishing_pos(exit_frame)
        if vanishing_pos is None:
            return

        self.rejected_vanish_positions.append(vanishing_pos.copy())

        if len(self.rejected_vanish_positions) > self.MAX_REJECTION_HISTORY:
            self.rejected_vanish_positions = self.rejected_vanish_positions[-self.MAX_REJECTION_HISTORY:]

        logger.debug(f"[TFEX] Logged rejected vanish position: ({vanishing_pos[0]:.1f}, {vanishing_pos[1]:.1f})")

    def _auto_reject_check(self, exit_frame: int) -> bool:
        if len(self.rejected_vanish_positions) < self.VANISH_CLUSTER_THRESHOLD:
            return False
        
        candidate_vanish_pos = self._find_vanishing_pos(exit_frame)
        if candidate_vanish_pos is None:
            return False

        x, y = candidate_vanish_pos[0], candidate_vanish_pos[1]
        nearby_count = 0

        for rej_x, rej_y in self.rejected_vanish_positions:
            if np.sqrt((x - rej_x)**2 + (y - rej_y)**2) <= self.VANISH_CLUSTER_RADIUS:
                nearby_count += 1
                if nearby_count >= self.VANISH_CLUSTER_THRESHOLD:
                    return True
        
        return False

    def _find_vanishing_pos(self, exit_frame:int) -> np.ndarray | None:
        if self.inst_count_per_frame[exit_frame] != 1:
            return

        solo_inst = get_instances_on_current_frame(self.pred_data_array, exit_frame)[0]
        solo_pos = self.centroids[exit_frame, solo_inst]

        pre_pos_0 = self.centroids[exit_frame - 1, 0]
        pre_pos_1 = self.centroids[exit_frame - 1, 1]

        dist_0 = np.linalg.norm(pre_pos_0 - solo_pos)
        dist_1 = np.linalg.norm(pre_pos_1 - solo_pos)

        return pre_pos_1 if dist_0 < dist_1 else pre_pos_0
    
    def _find_returning_pos(self, return_frame:int) -> np.ndarray | None:
        if self.inst_count_per_frame[return_frame] != 2 or self.inst_count_per_frame[return_frame-1] != 1:
            return
        
        solo_inst = get_instances_on_current_frame(self.pred_data_array, return_frame-1)[0]
        solo_pos = self.centroids[return_frame-1, solo_inst]

        post_pos_0 = self.centroids[return_frame, 0]
        post_pos_1 = self.centroids[return_frame, 1]

        dist_0 = np.linalg.norm(post_pos_0 - solo_pos)
        dist_1 = np.linalg.norm(post_pos_1 - solo_pos)

        return post_pos_1 if dist_0 < dist_1 else post_pos_0

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
        for inst_idx in range(2):
            in_zone_status[inst_idx] = self._in_zone_check_pos(self.centroids[frame_idx, inst_idx])
        return in_zone_status
    
    def _in_zone_check_pos(self, pos:np.ndarray) -> bool:
        xmin, ymin, xmax, ymax = self.exit_zone
        x, y = pos
        return (xmin <= x <= xmax) and (ymin <= y <= ymax)

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
        else:
            return 1e6

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

    def _validate_at_exit(self, exit_start: int) -> bool:
        if exit_start == 0:
            return False

        curr_valid = ~np.isnan(self.centroids[exit_start, :, 0])

        if not np.sum(curr_valid) == 1:
            return False

        vanished_idx = int(np.where(~curr_valid)[0][0])
        return vanished_idx == self.persistent_idx

    def _review_ambiguous_frames(self, ambiguous_frames: List[int], exit_start: int):
        candidates = sorted(set([f for f in ambiguous_frames if f < exit_start]))

        seg_start = get_prev_frame_in_list(self.exit_ends, exit_start-2)
        event_start_idx = seg_start - 10 if seg_start is not None else exit_start - 20
        swap_orders = self._launch_dialog_swap(
            ambiguous_list=candidates,
            event_start_idx=event_start_idx,
            event_end_idx=exit_start+11
            )

        if swap_orders:
            for frame_idx, order in swap_orders:
                if order == "i":
                    self._swap_ids_in_frame(frame_idx)
                else:
                    self._bulk_swap_from_frame(frame_idx)

    def _get_vanished_idx_at(self, exit_start: int) -> int:
        curr_valid = ~np.isnan(self.centroids[exit_start, :, 0])
        if np.sum(curr_valid) == 1:
            return int(np.where(~curr_valid)[0][0])
        return -1
    
    def _bulk_swap_from_frame(self, start_frame: int):
        logger.debug(f"[BK] Bulk swapping IDs from frame {start_frame} to end")
        self.pred_data_array[start_frame:] = self.pred_data_array[start_frame:, ::-1]
        self.centroids[start_frame:] = self.centroids[start_frame:, ::-1]
        self.last_known_pos = self.last_known_pos[::-1]
        self.kalman_filters.reverse()
        self.kalman_failure_count.reverse()

    def _launch_dialog_exit_return(
            self,
            frame_idx: int,
            mode:Literal["exit", "return"],
            event_start_idx=None,
            event_end_idx=None
            ) -> Tuple[bool, int]:
        if self.avtomat:
            return False, frame_idx

        last_frame_shown = self.last_exit_frame_shown if mode == "exit" else self.last_return_frame_shown
        frames_since_last = frame_idx - last_frame_shown
        if frames_since_last < self.USER_DIALOG_COOLDOWN:
            logger.info(
                f"[DIALOG] Cooldown active for {mode} at frame {frame_idx} (last: {last_frame_shown}, cooldown: {self.USER_DIALOG_COOLDOWN}."
            )
            return False, frame_idx
    
        dialog = Exit_Reentry_Dialog(
            dlc_data=self.dlc_data,
            extractor=self.extractor,
            pred_data_array=self.pred_data_array,
            current_frame_idx=frame_idx,
            exit_mode=mode=="exit",
            event_start_idx=event_start_idx,
            event_end_idx=event_end_idx,
            parent=self.main
        )
        dialog.exec()

        decision, confirmed_frame = dialog.user_decision

        match mode:
            case "exit": self.last_exit_frame_shown = frame_idx
            case "return": self.last_return_frame_shown = frame_idx

        return decision, confirmed_frame

    def _launch_dialog_swap(
            self,
            ambiguous_list:List[int],
            event_start_idx=None,
            event_end_idx=None
            ) -> List[Tuple[int, str]] | str:
        if self.avtomat and ambiguous_list:
            return [(ambiguous_list[0], "t")]

        dialog = Swap_Correction_Dialog(
            dlc_data=self.dlc_data,
            extractor=self.extractor,
            pred_data_array=self.pred_data_array,
            ambiguous_frames=ambiguous_list,
            event_start_idx=event_start_idx,
            event_end_idx=event_end_idx,
            parent=self.main
        )
        dialog.exec()

        if dialog.cancelled:
            return "exit"
        else:
            return dialog.user_decisions


class Track_Fixer_No_Exit(Track_Fixer):
    def __init__(
            self,
            pred_data_array:np.ndarray,
            dlc_data:Loaded_DLC_Data,
            extractor:Frame_Extractor,
            anglemap:Dict[str, int],
            blob_config:Optional[Blob_Config]=None,
            max_epochs:int=30,
            warmup_epochs:int=5,
            avtomat:bool = False,
            parent=None
            ):
        


        super().__init__(pred_data_array, 0, None, dlc_data, extractor, avtomat, parent)
        self.anglemap = anglemap
        self.epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.blob_config = blob_config
        self.eligible_frames = []

        self.temp_dir = os.path.join(
            self.extractor.get_video_dir(), "bvt_temp", self.extractor.get_video_name(no_ext=True))
        os.makedirs(self.temp_dir, exist_ok=True)

    def track_correction(self, start_idx = 0, end_idx = -1): 
        if end_idx == -1:
            end_idx = self.pred_data_array.shape[0]
            print(end_idx)

        ambiguous_frames = []
        for f in range(start_idx, end_idx):
            success = self._correct_frame_with_hungarian(f)
            if not success:
                ambiguous_frames.append(f)
                logger.debug(f"[TF] Ambiguous match, added to backtrack list")

        self._find_eligible_frames()

        eligible_in_range = [f for f in self.eligible_frames if f >= start_idx and f < end_idx]
        self.eligible_frames = eligible_in_range

        self._run_contrain_magic(ambiguous_frames)

        return self.pred_data_array

    def _find_eligible_frames(self, inst_dist_threshold:float=1.2, twist_angle_threshold:float=10.0):
        full_mask = np.zeros(self.total_frames, dtype=bool)

        instance_count = get_instance_count_per_frame(self.pred_data_array)
        two_inst_mask = instance_count==2

        if not np.any(two_inst_mask):
            return np.zeros(self.total_frames, dtype=bool)

        two_inst_array = self.pred_data_array[two_inst_mask]
        centroids_two_inst = self.centroids[two_inst_mask]

        head_idx = self.anglemap["head_idx"]
        all_head = self.pred_data_array[..., head_idx*3:head_idx*3+2]

        tail_idx = self.anglemap["tail_idx"]
        all_tail = self.pred_data_array[..., tail_idx*3:tail_idx*3+2]

        mice_length = np.nanmedian(np.linalg.norm(all_head - all_tail, axis=-1))

        dists = np.linalg.norm(centroids_two_inst[:, 1, :] - centroids_two_inst[:, 0, :], axis=-1)
        dist_mask = dists > inst_dist_threshold * mice_length

        outlier_twist = outlier_rotation(two_inst_array, self.anglemap, twist_angle_threshold)
        angle_mask = ~np.any(outlier_twist, axis=-1)

        full_mask[two_inst_mask] = dist_mask & angle_mask

        self.eligible_frames = np.where(full_mask)[0].tolist()
    
    def _crop_rotate_and_export(self):
        
        from core.io import Cutout_Exporter

        I = self.pred_data_array.shape[1]
        angles = np.zeros((self.total_frames, I))

        crop_coords = calculate_pose_array_bbox(self.pred_data_array[self.eligible_frames], padding=15).astype(np.uint16)
        cutout_dim = int(np.percentile(crop_coords[..., 2:4] - crop_coords[..., 0:2], 90))
        angles[self.eligible_frames] = np.rad2deg(calculate_pose_array_rotations(self.pred_data_array[self.eligible_frames], self.anglemap)) # (F, I)
        progress = Progress_Indicator_Dialog(0, 100, "Cutout Extraction", "Extracting cutouts from video", parent=self.main)

        co = Cutout_Exporter(
            save_folder=self.temp_dir,
            video_filepath=self.extractor.get_video_filepath(),
            frame_list=self.eligible_frames,
            centroids=self.centroids,
            cutout_dim=cutout_dim,
            angle_array=angles,
            grayscaling=True,
            blob_config=None,
            progress_callback=progress,
        )
        co.extract_frame()

    def _run_contrain_magic(self, ambiguous_frames:List[int]):
        
        from utils.embedding import Crop_Dataset, Embedding_Visualizer, Contrastive_Trainer

        embedding_filepath = os.path.join(self.temp_dir, 'embeddings.npz')
        if os.path.isfile(embedding_filepath):
            logger.info(f"Auto loading embedding file at {embedding_filepath}")
            cache = np.load(embedding_filepath)
            embeddings = cache['embeddings']
            motion_ids = cache['motion_ids'].tolist()
            frame_indices = cache['frame_indices'].tolist()
        else:

            from core.io import Cutout_Dataloader

            self._crop_rotate_and_export()
            loader = Cutout_Dataloader(self.temp_dir)
            crops, motion_ids, frame_indices, _, _ = loader.load_paired_tracks(n_mice=2)
            if len(crops) == 0:
                raise FileNotFoundError("[TF] No crops loaded. Check your folder path and filename format.")
            cds = Crop_Dataset(crops, motion_ids, frame_indices)
            trainer = Contrastive_Trainer()
            trainer.train(cds, epochs=self.epochs, warmup_epochs=self.warmup_epochs)
            embeddings = trainer.extract_embeddings(cds)
            np.savez(embedding_filepath, embeddings=embeddings,
                     motion_ids=motion_ids, frame_indices=frame_indices)

        kmeans = KMeans(n_clusters=2, random_state=42)
        visual_labels = kmeans.fit_predict(embeddings)
        
        vis = Embedding_Visualizer(embeddings, motion_ids, frame_indices, visual_labels)
        tsne_pix = vis.plot_tsne_combined()

        agreement_pix, stable_swap_candidates = vis.plot_agreement_timeline()

        tsne_pix.save(os.path.join(self.temp_dir, "tsne_combined.png"))
        agreement_pix.save(os.path.join(self.temp_dir, "agreement_timeline.png"))

        stable_spans = indices_to_spans(stable_swap_candidates)

        for start, end in stable_spans:
            if (start == 0) or (end == self.total_frames - 1):
                continue

            amb_in_range = [f for f in ambiguous_frames if f >= start and f < end]

            if self.avtomat:
                swap_orders = []
                if not amb_in_range:
                    swap_orders.append((start, "t"))
                    swap_orders.append((end, "t"))
                else:
                    swap_orders.append((amb_in_range[0], "t"))
                    if len(amb_in_range) >= 2:
                        swap_orders.append((amb_in_range[-1], "t"))
                    else:
                        swap_orders.append((end, "t"))
            else:
               swap_orders = self._launch_dialog_swap(amb_in_range, start-10, end+11)

            if swap_orders == "exit":
                return
            else:
                for frame_idx, order in swap_orders:
                    if order == "i":
                        self.pred_data_array = swap_track(self.pred_data_array, frame_idx)
                    else:
                        self.pred_data_array = swap_track(
                            self.pred_data_array, frame_idx, swap_range=list(range(frame_idx, end+11)))

        self.centroids, _ = calculate_pose_centroids(self.pred_data_array)
        if os.path.isfile(embedding_filepath):
            os.remove(embedding_filepath)