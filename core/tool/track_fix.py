import numpy as np
from typing import Tuple, List, Optional

from utils.pose import calculate_pose_centroids
from utils.helper import get_instance_count_per_frame
from utils.logger import logger

from .reviewer import Track_Correction_Dialog
from .mark_nav import get_prev_frame_in_list
from core.io import Frame_Extractor
from utils.dataclass import Loaded_DLC_Data


class Track_Fixer:
    KALMAN_MAX_ERROR = 80.0
    AMBIGUITY_THRESHOLD = 20.0
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
        self.last_valid_centroid = np.full((2, 2), np.nan)
        self.kalman_failure_count = [0, 0]

        self.exit_windows = np.zeros((self.total_frames,), dtype=bool)
        self.exit_starts = []

        self.ambiguous_frames: List[int] = []

        self._initialize_kalman_and_last_valid()

    def _initialize_kalman_and_last_valid(self):
        for inst_id in [0, 1]:
            for f in range(self.total_frames):
                if np.isnan(self.centroids[f, inst_id]).any():
                    continue

                x, y = self.centroids[f, inst_id]
                self.last_valid_centroid[inst_id] = [x, y]
                self.kalman_filters[inst_id] = Kalman(initial_state=[x, y, 0.0, 0.0])
                break

    def track_correction(self, start_idx: int = 0, end_idx: int = -1) -> np.ndarray:
        if end_idx == -1:
            end_idx = self.total_frames

        logger.debug(f"[TF] Starting track correction from frame {start_idx} to {end_idx}")
        
        self._detect_exit_windows(start_idx, end_idx)
        self.exit_starts = sorted(np.where((self.exit_windows[:-1] == False) & (self.exit_windows[1:] == True))[0] + 1)
        logger.debug(f"[TF] All non exit window frames: \n {sorted(np.where(~self.exit_windows)[0])}")

        self.ambiguous_frames = []
        for f in range(start_idx, end_idx):
            logger.debug(f"---------------- Frame {f} ----------------")
            self._enforce_exit_window_constraints(f)

            if self.exit_windows[f]:
                self._update_kalman_during_exit(f)
                logger.debug(f"[TF] Frame {f}: In exit window, updated Kalman for present mouse")
            else:
                success = self._correct_frame_with_hungarian(f)
                if not success:
                    self.ambiguous_frames.append(f)
                    logger.debug(f"[TF] Frame {f}: Ambiguous match, added to backtrack list")

            if f in self.exit_starts:
                logger.debug(f"[TF] Frame {f}: Detected exit start, validating exited mouse")
                self._validate_exit_and_backtrack_if_needed(f)

        logger.debug("Track correction completed.")
        return self.pred_data_array

    def _detect_exit_windows(self, start_idx: int, end_idx: int):
        """Detect exit windows; pause for user review on ambiguity."""
        logger.debug("Pass 1: Starting exit window detection with human review")
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

        xmin, ymin, xmax, ymax = self.exit_zone

        i = 0
        while i < len(raw_segments):
            curr_start, curr_end, curr_val = raw_segments[i]
            if curr_val != 1:
                i += 1
                continue

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
                else:
                    logger.debug(f"min_cost tooo large: {min_cost}")

            exit_frame = curr_start
            pre_frame = exit_frame - 1
            valid_exit = self._is_valid_exit_at(pre_frame, exit_frame, xmin, ymin, xmax, ymax)

            if not valid_exit:
                logger.debug(f"[TFEX] Exit at frame {exit_frame} failed validation, requesting user review")
                decision, confirmed_frame = self._launch_dialog(exit_frame, "exit")
                if not decision:
                    logger.debug(f"[TFEX] User rejected exit at frame {exit_frame}, skipping window")
                    i += 1
                    continue
                exit_frame = confirmed_frame
                if exit_frame <= 0 or exit_frame >= end_idx:
                    i += 1
                    continue
                curr_start = exit_frame
                logger.debug(f"[TFEX] User confirmed exit at frame {exit_frame}")

            window_accepted = False
            j = i + 1

            if j >= len(raw_segments):
                self.exit_windows[curr_start:curr_end+1] = True
                logger.debug(f"[TFEX] Accepted open-ended exit window: ({curr_start}, {curr_end})")
                window_accepted = True
            else:
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

                        if self._handle_single_observation(actual_ext_frame, ref_positions):
                            logger.debug(
                                f"[TFEX] Short re-entry segment ({return_frame}-{exit_again_frame}, duration={exit_again_frame - return_frame + 1}) "
                                f"rejected as spurious detection: frame {exit_again_frame + 1} consistent with post-return state"
                            )
                            j += 1
                            continue

                    valid_return = self._is_valid_return_at(during_frame, return_frame, xmin, ymin, xmax, ymax)
                    if not valid_return:
                        logger.debug(f"[TFEX] Return candidate at frame {return_frame} failed validation, requesting user review")
                        decision, confirmed_frame = self._launch_dialog(return_frame, "return", curr_start)
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

    def _is_valid_exit_at(self, pre_frame: int, exit_frame: int, xmin, ymin, xmax, ymax) -> bool:
        if (self.inst_count_per_frame[pre_frame] != 2 or 
            self.inst_count_per_frame[exit_frame] not in (0, 1)):
            return False
        pre_centroids = self.centroids[pre_frame]
        exit_centroids = self.centroids[exit_frame]
        valid_exit = ~np.isnan(exit_centroids[:, 0])
        n_exit = np.sum(valid_exit)
        if n_exit != 1:
            return False
        
        in_zone_any = False
        for inst_idx in range(2):
            x, y = pre_centroids[inst_idx]
            in_zone = (xmin <= x <= xmax) and (ymin <= y <= ymax)
            logger.debug(f"[TFEX] Exit validation for Inst {inst_idx} at {exit_frame}: pos=({x:.1f},{y:.1f}), in_zone={in_zone}")
            in_zone_any = True if in_zone else in_zone_any
        return in_zone_any

    def _is_valid_return_at(self, during_frame: int, return_frame: int, xmin, ymin, xmax, ymax) -> bool:
        if (self.inst_count_per_frame[during_frame] not in (0, 1) or
            self.inst_count_per_frame[return_frame] != 2):
            return False
        during_centroids = self.centroids[during_frame]
        post_centroids = self.centroids[return_frame]
        valid_during = ~np.isnan(during_centroids[:, 0])
        n_during = np.sum(valid_during)
        if n_during != 1:
            return False
        
        in_zone_any = False
        for inst_idx in range(2):
            x, y = post_centroids[inst_idx]
            in_zone = (xmin <= x <= xmax) and (ymin <= y <= ymax)
            logger.debug(f"[TFEX] Return validation for Inst {inst_idx} at {return_frame}: pos=({x:.1f},{y:.1f}), in_zone={in_zone}")
            in_zone_any = True if in_zone else in_zone_any
        return in_zone_any

    def _enforce_exit_window_constraints(self, frame_idx: int):
        if not self.exit_windows[frame_idx]:
            return
        if frame_idx in self.exit_starts:
            return

        start = get_prev_frame_in_list(self.exit_starts, frame_idx)
        exit_valid = ~np.isnan(self.centroids[start, :, 0])
        if not np.sum(exit_valid) == 1:
            return

        exited_id = int(np.where(~exit_valid)[0][0])
        curr_valid = ~np.isnan(self.centroids[frame_idx, :, 0])
        n_curr = np.sum(curr_valid)

        if n_curr == 2:
            self._clear_instance_at_frame(frame_idx, exited_id)
            logger.debug(f"[TFEX] Frame {frame_idx}: Cleared spurious detection of exited mouse {exited_id}")
        elif n_curr == 1:
            visible_id = int(np.where(curr_valid)[0][0])
            if visible_id == exited_id:
                self._swap_ids_in_frame(frame_idx)
                logger.debug(f"[TFEX] Frame {frame_idx}: Swapped IDs because exited mouse {exited_id} was visible")

    def _clear_instance_at_frame(self, frame_idx: int, inst_id: int):
        self.pred_data_array[frame_idx, inst_id, :] = np.nan
        self.centroids[frame_idx, inst_id, :] = np.nan
        self.inst_count_per_frame[frame_idx] = np.sum(~np.isnan(self.centroids[frame_idx, :, 0]))

    def _swap_ids_in_frame(self, frame_idx: int):
        self.pred_data_array[frame_idx] = self.pred_data_array[frame_idx, ::-1]
        self.centroids[frame_idx] = self.centroids[frame_idx, ::-1]

    def _update_kalman_during_exit(self, frame_idx: int):
        valid_obs = ~np.isnan(self.centroids[frame_idx, :, 0])
        for inst_id in [0, 1]:
            if valid_obs[inst_id]:
                z = self.centroids[frame_idx, inst_id]
                if self.kalman_filters[inst_id] is None:
                    self.kalman_filters[inst_id] = Kalman(initial_state=[z[0], z[1], 0.0, 0.0])
                    self.kalman_failure_count[inst_id] = 0
                else:
                    self.kalman_filters[inst_id].update(z)

    def _correct_frame_with_hungarian(self, frame_idx: int) -> bool:
        current_centroids = self.centroids[frame_idx]
        valid_obs = ~np.isnan(current_centroids[:, 0])
        n_obs = np.sum(valid_obs)

        if n_obs == 0:
            return True

        for inst_id in [0, 1]:
            if valid_obs[inst_id]:
                self.last_valid_centroid[inst_id] = current_centroids[inst_id].copy()

        ref_positions = self._get_reference_positions(frame_idx)

        if n_obs == 1:
            result = self._handle_single_observation(frame_idx, ref_positions)
            logger.debug(f"Frame {frame_idx}: Single observation handled, success={result}")
            return result
        elif n_obs == 2:
            result = self._handle_two_observations(frame_idx, ref_positions)
            logger.debug(f"Frame {frame_idx}: Two observations handled, success={result}")
            return result
        return True

    def _get_reference_positions(self, frame_idx: int) -> np.ndarray:
        current_obs = self.centroids[frame_idx]
        valid_obs = ~np.isnan(current_obs[:, 0])
        ref_positions = np.full((2, 2), np.nan)

        for inst_id in [0, 1]:
            logger.debug(f"  → Processing instance {inst_id}")

            kalman_pred = None
            if self.kalman_filters[inst_id] is not None:
                state = self.kalman_filters[inst_id].predict()
                kalman_pred = state[:2]
                logger.debug(f"    Kalman prediction for inst {inst_id}: {kalman_pred}")
            else:
                logger.debug(f"    No active Kalman filter for inst {inst_id}")

            kalman_trusted = False
            if kalman_pred is not None and not np.isnan(kalman_pred).any() and np.any(valid_obs):
                distances = []
                for obs_id in range(2):
                    if valid_obs[obs_id]:
                        dist = np.linalg.norm(kalman_pred - current_obs[obs_id])
                        distances.append(dist)
                        logger.debug(f"      Distance to obs {obs_id} ({current_obs[obs_id]}): {dist:.2f}")
                
                min_dist = min(distances)
                logger.debug(f"    Min distance from Kalman pred to any valid obs: {min_dist:.2f}")
                
                if min_dist <= self.KALMAN_MAX_ERROR:
                    kalman_trusted = True
                    logger.debug(f"    Kalman prediction TRUSTED (≤ {self.KALMAN_MAX_ERROR})")
                else:
                    logger.debug(f"    Kalman prediction NOT trusted (exceeds max error {self.KALMAN_MAX_ERROR})")

            if kalman_trusted:
                ref_positions[inst_id] = kalman_pred
                self.kalman_failure_count[inst_id] = 0
                logger.debug(f"    Using Kalman prediction as reference for inst {inst_id}: {kalman_pred}")
            else:
                if not np.isnan(self.last_valid_centroid[inst_id]).any():
                    ref_positions[inst_id] = self.last_valid_centroid[inst_id]
                    logger.debug(f"    Fallback to last valid centroid for inst {inst_id}: {self.last_valid_centroid[inst_id]}")
                else:
                    logger.debug(f"    No valid fallback for inst {inst_id}; reference remains NaN")

                if self.kalman_filters[inst_id] is not None:
                    self.kalman_failure_count[inst_id] += 1
                    logger.debug(f"    Kalman failure count for inst {inst_id}: {self.kalman_failure_count[inst_id]}")
                    if self.kalman_failure_count[inst_id] >= self.KALMAN_RESET_THRESHOLD:
                        self.kalman_filters[inst_id] = None
                        logger.warning(f"    Kalman filter for inst {inst_id} RESET due to {self.kalman_failure_count[inst_id]} consecutive failures")

        logger.debug(f"[Frame {frame_idx}] Final ref_positions:\n{ref_positions}")
        return ref_positions

    def _handle_single_observation(self, frame_idx: int, ref_positions: np.ndarray) -> bool:
        obs_id = int(np.where(~np.isnan(self.centroids[frame_idx, :, 0]))[0][0])
        obs_pos = self.centroids[frame_idx, obs_id]

        costs = []
        candidates = []
        for candidate_id in [0, 1]:
            if not np.isnan(ref_positions[candidate_id, 0]):
                d = np.linalg.norm(obs_pos - ref_positions[candidate_id])
                costs.append(d)
                candidates.append(candidate_id)

        if not candidates:
            return False

        if len(candidates) == 1:
            assigned_id = candidates[0]
            if assigned_id != obs_id:
                self._swap_ids_in_frame(frame_idx)
                logger.debug(f"Frame {frame_idx}: Single obs assigned to {assigned_id} (was {obs_id}), swapped")
            self._update_kalman_with_observation(frame_idx, [assigned_id], [assigned_id])
            cost = costs[0]
            if cost <= self.MAX_DIST_THRESHOLD:
                return True
            else:
                logger.debug(f"Frame {frame_idx}: Single obs assignment cost too large: (cost: {cost:.1f} > {self.MAX_DIST_THRESHOLD})")
                return False
        else:
            c0, c1 = costs
            if abs(c0 - c1) < self.AMBIGUITY_THRESHOLD:
                logger.debug(f"Frame {frame_idx}: Ambiguous single observation (costs {c0:.1f}, {c1:.1f})")
                return False
            else:
                winner = candidates[0] if c0 < c1 else candidates[1]
                if winner != obs_id:
                    self._swap_ids_in_frame(frame_idx)
                    logger.debug(f"Frame {frame_idx}: Single obs clear winner {winner} (was {obs_id}), swapped")
                self._update_kalman_with_observation(frame_idx, [winner], [winner])
                return True

    def _handle_two_observations(self, frame_idx:int, ref_positions:np.ndarray, get_min_cost:bool=False) -> bool | float:
        cost_matrix = np.full((2, 2), np.inf)
        for obs_id in [0, 1]:
            for candidate_id in [0, 1]:
                if not np.isnan(ref_positions[candidate_id, 0]):
                    d = np.linalg.norm(
                        self.centroids[frame_idx, obs_id] - ref_positions[candidate_id]
                    )
                    cost_matrix[obs_id, candidate_id] = d

        if np.all(np.isinf(cost_matrix)):
            return False

        cost_matrix_lap = np.where(np.isinf(cost_matrix), 1e6, cost_matrix)

        cost_swap = cost_matrix_lap[0, 1] + cost_matrix_lap[1, 0]
        cost_no_swap = cost_matrix_lap[0, 0] + cost_matrix_lap[1, 1]

        if get_min_cost:
            return min(cost_swap, cost_no_swap)

        if abs(cost_swap - cost_no_swap) < self.AMBIGUITY_THRESHOLD:
            logger.debug(f"Frame {frame_idx}: Ambiguous two-obs match (Δcost={abs(cost_swap - cost_no_swap):.1f})")
            return False

        needs_swap = cost_swap < cost_no_swap
        if needs_swap:
            self._swap_ids_in_frame(frame_idx)
            logger.debug(f"Frame {frame_idx}: Two-obs clear swap needed (cost_swap={cost_swap:.1f} < no_swap={cost_no_swap:.1f})")

        self._update_kalman_with_observation(frame_idx, [0, 1], [0, 1])
        return True

    def _update_kalman_with_observation(self, frame_idx: int, observed_inst_ids: List[int], assigned_ids: List[int]):
        for obs_id, global_id in zip(observed_inst_ids, assigned_ids):
            z = self.centroids[frame_idx, obs_id]
            if np.isnan(z).any():
                continue
            if self.kalman_filters[global_id] is None:
                self.kalman_filters[global_id] = Kalman(initial_state=[z[0], z[1], 0.0, 0.0])
                self.kalman_failure_count[global_id] = 0
            else:
                self.kalman_filters[global_id].update(z)

    def _validate_exit_and_backtrack_if_needed(self, exit_start: int):
        if exit_start == 0:
            return

        curr_valid = ~np.isnan(self.centroids[exit_start, :, 0])

        if not np.sum(curr_valid) == 1:
            return

        vanished_id = int(np.where(~curr_valid)[0][0])
        if vanished_id == self.persistent_idx:
            logger.warning(f"Frame {exit_start}: Persistent mouse (ID {self.persistent_idx}) appears to have exited! Triggering backtracking.")
            self._backtrack_ambiguous_frames(exit_start)

    def _backtrack_ambiguous_frames(self, exit_start: int):
        candidates = [f for f in self.ambiguous_frames if f < exit_start]
        candidates.sort(reverse=True)
        logger.debug(f"Backtracking from exit start {exit_start} with candidate frames: {candidates}")

        for frame in candidates:
            logger.debug(f"Reviewing ambiguous frame {frame} for potential swap")
            decision, confirmed_frame = self._launch_dialog(frame, "swap")
            if decision:
                logger.info(f"User confirmed swap at frame {confirmed_frame}, applying bulk swap")
                self._bulk_swap_from_frame(confirmed_frame)
                new_vanished = self._get_vanished_id_at(exit_start)
                if new_vanished != self.persistent_idx:
                    logger.info(f"Backtracking successful: exited mouse is now ID {new_vanished} (not persistent)")
                    return
                else:
                    logger.warning(f"Backtracking at frame {confirmed_frame} did not resolve issue, trying earlier frame")
        logger.error(f"Backtracking failed for exit at {exit_start}: persistent mouse still appears to exit")

    def _get_vanished_id_at(self, exit_start: int) -> int:
        pre_valid = ~np.isnan(self.centroids[exit_start - 1, :, 0])
        curr_valid = ~np.isnan(self.centroids[exit_start, :, 0])
        if pre_valid.all() and np.sum(curr_valid) == 1:
            return int(np.where(~curr_valid)[0][0])
        return -1
    
    def _bulk_swap_from_frame(self, start_frame: int):
        """Flip IDs for all frames from start_frame to end."""
        logger.debug(f"Bulk swapping IDs from frame {start_frame} to end")
        self.pred_data_array[start_frame:] = self.pred_data_array[start_frame:, ::-1]
        self.centroids[start_frame:] = self.centroids[start_frame:, ::-1]
        self.kalman_filters = [None, None]
        self.kalman_failure_count = [0, 0]
        self._initialize_kalman_and_last_valid()

    def _launch_dialog(self, frame_idx: int, mode: str, last_event_idx:Optional[int]=None) -> Tuple[bool, int]:
        dialog = Track_Correction_Dialog(
            dlc_data=self.dlc_data,
            extractor=self.extractor,
            pred_data_array=self.pred_data_array,
            current_frame_idx=frame_idx,
            mode=mode,
            last_event_idx=last_event_idx,
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