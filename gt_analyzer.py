"""
Ground Truth Analyzer: Generate Visual Reports from Annotation Data

Reads the ground_truth_*.json files produced by the annotator and generates:
- Confidence distribution plots
- Area ratio analysis  
- Y-position spawn zone mapping
- Movement pattern visualization
- Recommended threshold values with confidence intervals

Usage:
    python gt_analyzer.py ground_truth_20260211_143000.json
    python gt_analyzer.py ground_truth_*.json  # Analyze multiple files

Author: Ahmad's Mosque Attendance System
"""

import json
import sys
import os
import math
from pathlib import Path
from datetime import datetime

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")
    print("Text-only analysis will be shown.\n")


def load_ground_truth(filepath):
    """Load and validate a ground truth JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data


def analyze_tracks(data):
    """Deep analysis of labeled tracks."""
    tracks = data.get("tracks", {})
    
    in_tracks = {k: v for k, v in tracks.items() if v.get("label") == "IN"}
    out_tracks = {k: v for k, v in tracks.items() if v.get("label") == "OUT"}
    unlabeled = {k: v for k, v in tracks.items() if v.get("label") is None}
    
    results = {
        "counts": {
            "total": len(tracks),
            "in": len(in_tracks),
            "out": len(out_tracks),
            "unlabeled": len(unlabeled),
        }
    }
    
    # === Confidence Analysis ===
    all_labeled = {**in_tracks, **out_tracks}
    
    if all_labeled:
        # Per-frame confidence for all labeled tracks
        in_all_confs = []
        out_all_confs = []
        for t in in_tracks.values():
            in_all_confs.extend(t.get("confidence_history", []))
        for t in out_tracks.values():
            out_all_confs.extend(t.get("confidence_history", []))
        
        # Max confidence per track
        in_max_confs = [t["max_confidence"] for t in in_tracks.values()]
        out_max_confs = [t["max_confidence"] for t in out_tracks.values()]
        unlabeled_max_confs = [t["max_confidence"] for t in unlabeled.values()]
        
        all_max = in_max_confs + out_max_confs
        
        results["confidence"] = {
            "in_max_confs": in_max_confs,
            "out_max_confs": out_max_confs,
            "unlabeled_max_confs": unlabeled_max_confs,
            "in_frame_confs": in_all_confs,
            "out_frame_confs": out_all_confs,
            "all_max": all_max,
        }
        
        # Sweep confidence thresholds to find optimal
        conf_sweep = {}
        for thresh in [i * 0.05 for i in range(1, 20)]:
            caught_in = sum(1 for c in in_max_confs if c >= thresh)
            caught_out = sum(1 for c in out_max_confs if c >= thresh)
            false_pos = sum(1 for c in unlabeled_max_confs if c >= thresh)
            total_caught = caught_in + caught_out
            total_labeled = len(in_max_confs) + len(out_max_confs)
            recall = total_caught / total_labeled if total_labeled > 0 else 0
            conf_sweep[round(thresh, 2)] = {
                "recall": round(recall, 3),
                "caught": total_caught,
                "missed": total_labeled - total_caught,
                "false_positives": false_pos,
            }
        results["confidence_sweep"] = conf_sweep
    
    # === Area Analysis ===
    in_area_ratios = []
    out_area_ratios = []
    for t in in_tracks.values():
        hist = t.get("area_history", [])
        if len(hist) >= 2 and hist[0] > 0:
            in_area_ratios.append(hist[-1] / hist[0])
    for t in out_tracks.values():
        hist = t.get("area_history", [])
        if len(hist) >= 2 and hist[0] > 0:
            out_area_ratios.append(hist[-1] / hist[0])
    
    results["area"] = {
        "in_ratios": in_area_ratios,
        "out_ratios": out_area_ratios,
    }
    
    # === Y Position Analysis ===
    in_y_first = [t["first_centroid_y"] for t in in_tracks.values() if t.get("first_centroid_y") is not None]
    in_y_last = [t["last_centroid_y"] for t in in_tracks.values() if t.get("last_centroid_y") is not None]
    out_y_first = [t["first_centroid_y"] for t in out_tracks.values() if t.get("first_centroid_y") is not None]
    out_y_last = [t["last_centroid_y"] for t in out_tracks.values() if t.get("last_centroid_y") is not None]
    
    results["y_positions"] = {
        "in_first": in_y_first,
        "in_last": in_y_last,
        "out_first": out_y_first,
        "out_last": out_y_last,
    }
    
    # === Movement Trajectories (full 2D paths) ===
    in_trajectories = []
    out_trajectories = []
    for t in in_tracks.values():
        y_hist = t.get("centroid_y_history", [])
        x_hist = t.get("centroid_x_history", [])
        a_hist = t.get("area_history", [])
        path = t.get("path", [])
        traj = {
            "y": y_hist, "x": x_hist, "area": a_hist, "id": t["track_id"],
            "path": path,
            "path_length": t.get("path_length", 0),
            "displacement": t.get("displacement", 0),
            "linearity": t.get("linearity", 0),
            "direction_angle": t.get("direction_angle"),
            "avg_speed": t.get("avg_speed", 0),
            "frame_numbers": t.get("frame_numbers", []),
            "bbox_history": t.get("bbox_history", []),
        }
        if y_hist:
            in_trajectories.append(traj)
    for t in out_tracks.values():
        y_hist = t.get("centroid_y_history", [])
        x_hist = t.get("centroid_x_history", [])
        a_hist = t.get("area_history", [])
        path = t.get("path", [])
        traj = {
            "y": y_hist, "x": x_hist, "area": a_hist, "id": t["track_id"],
            "path": path,
            "path_length": t.get("path_length", 0),
            "displacement": t.get("displacement", 0),
            "linearity": t.get("linearity", 0),
            "direction_angle": t.get("direction_angle"),
            "avg_speed": t.get("avg_speed", 0),
            "frame_numbers": t.get("frame_numbers", []),
            "bbox_history": t.get("bbox_history", []),
        }
        if y_hist:
            out_trajectories.append(traj)
    
    results["trajectories"] = {
        "in": in_trajectories,
        "out": out_trajectories,
    }
    
    # === Path Metrics Summary ===
    in_linearities = [t.get("linearity", 0) for t in in_tracks.values() if t.get("linearity")]
    out_linearities = [t.get("linearity", 0) for t in out_tracks.values() if t.get("linearity")]
    in_angles = [t.get("direction_angle") for t in in_tracks.values() if t.get("direction_angle") is not None]
    out_angles = [t.get("direction_angle") for t in out_tracks.values() if t.get("direction_angle") is not None]
    in_speeds = [t.get("avg_speed", 0) for t in in_tracks.values() if t.get("avg_speed", 0) > 0]
    out_speeds = [t.get("avg_speed", 0) for t in out_tracks.values() if t.get("avg_speed", 0) > 0]
    in_path_lengths = [t.get("path_length", 0) for t in in_tracks.values() if t.get("path_length", 0) > 0]
    out_path_lengths = [t.get("path_length", 0) for t in out_tracks.values() if t.get("path_length", 0) > 0]
    
    results["path_metrics"] = {
        "in_linearities": in_linearities,
        "out_linearities": out_linearities,
        "in_angles": in_angles,
        "out_angles": out_angles,
        "in_speeds": in_speeds,
        "out_speeds": out_speeds,
        "in_path_lengths": in_path_lengths,
        "out_path_lengths": out_path_lengths,
    }
    
    # === Optimal Counting Line Sweep ===
    # Test horizontal lines at every Y and see which gives best IN/OUT separation
    crossing_results = []
    for y_line in [i * 0.01 for i in range(20, 80)]:
        correct_in = 0  # IN tracks that cross from below threshold to above (approaching = y decreasing)
        correct_out = 0  # OUT tracks that cross from above to below (departing = y increasing)
        wrong_in = 0
        wrong_out = 0
        for t in in_tracks.values():
            y_hist = t.get("centroid_y_history", [])
            if len(y_hist) >= 2:
                crossed_down = any(y_hist[i] >= y_line and y_hist[i+1] < y_line for i in range(len(y_hist)-1))
                crossed_up = any(y_hist[i] < y_line and y_hist[i+1] >= y_line for i in range(len(y_hist)-1))
                if crossed_down:  # Moving up (IN = approaching from bottom)
                    correct_in += 1
                elif crossed_up:
                    wrong_in += 1
        for t in out_tracks.values():
            y_hist = t.get("centroid_y_history", [])
            if len(y_hist) >= 2:
                crossed_up = any(y_hist[i] < y_line and y_hist[i+1] >= y_line for i in range(len(y_hist)-1))
                crossed_down = any(y_hist[i] >= y_line and y_hist[i+1] < y_line for i in range(len(y_hist)-1))
                if crossed_up:  # Moving down (OUT = departing)
                    correct_out += 1
                elif crossed_down:
                    wrong_out += 1
        total_correct = correct_in + correct_out
        total_wrong = wrong_in + wrong_out
        total_labeled = len(in_tracks) + len(out_tracks)
        accuracy = total_correct / total_labeled if total_labeled > 0 else 0
        crossing_results.append({
            "y_line": round(y_line, 2),
            "accuracy": round(accuracy, 4),
            "correct_in": correct_in,
            "correct_out": correct_out,
            "wrong_in": wrong_in,
            "wrong_out": wrong_out,
        })
    
    results["crossing_line_sweep"] = crossing_results
    best_line = max(crossing_results, key=lambda r: r["accuracy"]) if crossing_results else None
    results["best_counting_line"] = best_line
    
    # === Track Duration ===
    in_durations = [t["total_frames_visible"] for t in in_tracks.values()]
    out_durations = [t["total_frames_visible"] for t in out_tracks.values()]
    unlabeled_durations = [t["total_frames_visible"] for t in unlabeled.values()]
    
    results["durations"] = {
        "in": in_durations,
        "out": out_durations,
        "unlabeled": unlabeled_durations,
    }
    
    # === X-Position Analysis ===
    in_x_first = [t.get("first_centroid_x") for t in in_tracks.values() if t.get("first_centroid_x") is not None]
    in_x_last = [t.get("last_centroid_x") for t in in_tracks.values() if t.get("last_centroid_x") is not None]
    out_x_first = [t.get("first_centroid_x") for t in out_tracks.values() if t.get("first_centroid_x") is not None]
    out_x_last = [t.get("last_centroid_x") for t in out_tracks.values() if t.get("last_centroid_x") is not None]
    
    results["x_positions"] = {
        "in_first": in_x_first,
        "in_last": in_x_last,
        "out_first": out_x_first,
        "out_last": out_x_last,
    }
    
    # === Track Overlap / Fragmentation Detection ===
    # Find tracks that might be the same person (close in position when one ends and another starts)
    all_labeled = {**in_tracks, **out_tracks}
    overlaps = []
    track_list = list(all_labeled.values())
    for i in range(len(track_list)):
        for j in range(i + 1, len(track_list)):
            t1 = track_list[i]
            t2 = track_list[j]
            # Check if t1 ends near where t2 starts (or vice versa)
            t1_last_y = t1.get("last_centroid_y")
            t1_last_x = t1.get("last_centroid_x", 0.5)
            t2_first_y = t2.get("first_centroid_y")
            t2_first_x = t2.get("first_centroid_x", 0.5)
            t1_end = t1.get("last_frame", 0)
            t2_start = t2.get("first_frame", 0)
            
            if t1_last_y is None or t2_first_y is None:
                continue
            
            # Check both directions
            for end_x, end_y, end_f, start_x, start_y, start_f, id_a, id_b in [
                (t1_last_x, t1_last_y, t1_end, t2_first_x, t2_first_y, t2_start, t1["track_id"], t2["track_id"]),
                (t2_first_x if t2.get("last_centroid_x") else 0.5, t2.get("last_centroid_y", 0), t2.get("last_frame", 0),
                 t1_last_x if t1.get("first_centroid_x") else 0.5, t1.get("first_centroid_y", 0), t1.get("first_frame", 0),
                 t2["track_id"], t1["track_id"]),
            ]:
                frame_gap = start_f - end_f
                if 0 < frame_gap < 30:  # Within 30 frames
                    dist = ((end_x - start_x)**2 + (end_y - start_y)**2)**0.5
                    if dist < 0.1:  # Within 10% of frame
                        overlaps.append({
                            "track_a": id_a,
                            "track_b": id_b,
                            "frame_gap": frame_gap,
                            "position_distance": round(dist, 4),
                            "label_a": t1.get("label"),
                            "label_b": t2.get("label"),
                        })
    
    results["potential_fragments"] = overlaps
    
    # === Heatmap data (start/end positions for all tracks) ===
    start_positions = {"in": [], "out": [], "unlabeled": []}
    end_positions = {"in": [], "out": [], "unlabeled": []}
    for t in in_tracks.values():
        x = t.get("first_centroid_x")
        y = t.get("first_centroid_y")
        if x is not None and y is not None:
            start_positions["in"].append((x, y))
        x = t.get("last_centroid_x")
        y = t.get("last_centroid_y")
        if x is not None and y is not None:
            end_positions["in"].append((x, y))
    for t in out_tracks.values():
        x = t.get("first_centroid_x")
        y = t.get("first_centroid_y")
        if x is not None and y is not None:
            start_positions["out"].append((x, y))
        x = t.get("last_centroid_x")
        y = t.get("last_centroid_y")
        if x is not None and y is not None:
            end_positions["out"].append((x, y))
    for t in unlabeled.values():
        x = t.get("first_centroid_x")
        y = t.get("first_centroid_y")
        if x is not None and y is not None:
            start_positions["unlabeled"].append((x, y))
        x = t.get("last_centroid_x")
        y = t.get("last_centroid_y")
        if x is not None and y is not None:
            end_positions["unlabeled"].append((x, y))
    
    results["heatmap"] = {
        "start_positions": start_positions,
        "end_positions": end_positions,
    }
    
    return results


def compute_recommendations(results, fps=30.0):
    """Compute recommended threshold values from analysis."""
    rec = {}
    
    # Confidence threshold
    if "confidence" in results:
        all_max = results["confidence"]["all_max"]
        if all_max:
            sorted_confs = sorted(all_max)
            p10 = sorted_confs[max(0, len(sorted_confs) // 10)]
            rec["DETECTION_CONFIDENCE"] = {
                "value": round(p10 * 0.85, 3),
                "reasoning": f"85% of 10th percentile ({p10:.3f}) of labeled tracks' max confidence",
                "range": f"{sorted_confs[0]:.3f} - {sorted_confs[-1]:.3f}"
            }
    
    # Area thresholds
    if "area" in results:
        in_r = results["area"]["in_ratios"]
        out_r = results["area"]["out_ratios"]
        if in_r:
            med_in = sorted(in_r)[len(in_r) // 2]
            rec["AREA_GROWTH_FOR_APPROACH"] = {
                "value": round(med_in * 0.85, 3),
                "reasoning": f"85% of median IN area ratio ({med_in:.3f})",
            }
        if out_r:
            med_out = sorted(out_r)[len(out_r) // 2]
            rec["AREA_SHRINK_FOR_DEPART"] = {
                "value": round(med_out * 1.15, 3),
                "reasoning": f"115% of median OUT area ratio ({med_out:.3f})",
            }
    
    # Y thresholds (spawn zones)
    yp = results.get("y_positions", {})
    in_first = yp.get("in_first", [])
    out_first = yp.get("out_first", [])
    in_last = yp.get("in_last", [])
    out_last = yp.get("out_last", [])
    
    if in_first:
        in_spawn = sorted(in_first)[len(in_first) // 2]
        rec["SPAWN_FAR_THRESHOLD"] = {
            "value": round(in_spawn, 3),
            "reasoning": f"Median first-seen Y for IN tracks",
        }
    if out_first:
        out_spawn = sorted(out_first)[len(out_first) // 2]
        rec["SPAWN_NEAR_THRESHOLD"] = {
            "value": round(out_spawn, 3),
            "reasoning": f"Median first-seen Y for OUT tracks",
        }
    
    # Crossing threshold
    if in_last and out_last:
        in_end = sum(in_last) / len(in_last)
        out_end = sum(out_last) / len(out_last)
        rec["NEAR_EDGE_THRESHOLD"] = {
            "value": round((in_end + out_end) / 2, 3),
            "reasoning": f"Midpoint between IN endpoint ({in_end:.3f}) and OUT endpoint ({out_end:.3f})",
        }
    
    # Best counting line from crossing sweep
    best_line = results.get("best_counting_line")
    if best_line:
        rec["BEST_COUNTING_LINE_Y"] = {
            "value": best_line["y_line"],
            "reasoning": f"Highest accuracy ({best_line['accuracy']*100:.1f}%) from line-crossing sweep: "
                         f"{best_line['correct_in']} IN + {best_line['correct_out']} OUT correct",
        }
    
    # Path linearity insight
    pm = results.get("path_metrics", {})
    in_lin = pm.get("in_linearities", [])
    out_lin = pm.get("out_linearities", [])
    if in_lin:
        avg_in_lin = sum(in_lin) / len(in_lin)
        rec["IN_PATH_LINEARITY"] = {
            "value": round(avg_in_lin, 3),
            "reasoning": f"Average linearity of IN paths (1.0=straight). "
                         f"{'Paths are straight - simple line crossing works' if avg_in_lin > 0.8 else 'Paths curve - consider multi-zone or trajectory-based counting'}",
        }
    if out_lin:
        avg_out_lin = sum(out_lin) / len(out_lin)
        rec["OUT_PATH_LINEARITY"] = {
            "value": round(avg_out_lin, 3),
            "reasoning": f"Average linearity of OUT paths (1.0=straight). "
                         f"{'Paths are straight - simple line crossing works' if avg_out_lin > 0.8 else 'Paths curve - consider multi-zone or trajectory-based counting'}",
        }
    
    # Min track duration
    durations = results.get("durations", {})
    all_dur = durations.get("in", []) + durations.get("out", [])
    if all_dur:
        min_dur = min(all_dur)
        rec["MIN_TRACK_DURATION_MS"] = {
            "value": round(min_dur / fps * 1000 * 0.8),
            "reasoning": f"80% of shortest labeled track ({min_dur} frames = {min_dur/fps*1000:.0f}ms)",
        }
    
    return rec


def plot_path_analysis(results, recommendations, output_path, frame_size=None, fps=30.0):
    """Generate dedicated path/trajectory analysis plots."""
    if not HAS_MATPLOTLIB:
        return
    
    traj = results.get("trajectories", {})
    in_traj = traj.get("in", [])
    out_traj = traj.get("out", [])
    
    if not in_traj and not out_traj:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle("Path & Trajectory Analysis - Finding Best Counting Strategy",
                 fontsize=14, fontweight='bold')
    
    # 1. Full 2D path overlay (bird's eye view of all paths)
    ax = axes[0, 0]
    for t in in_traj:
        x = t.get("x", [])
        y = t.get("y", [])
        if x and y:
            ax.plot(x, y, color='green', alpha=0.5, linewidth=1.5)
            ax.scatter([x[0]], [y[0]], color='green', marker='o', s=30, zorder=5)  # start
            ax.scatter([x[-1]], [y[-1]], color='green', marker='x', s=40, zorder=5)  # end
            ax.annotate(str(t["id"]), (x[0], y[0]), fontsize=6, color='green')
    for t in out_traj:
        x = t.get("x", [])
        y = t.get("y", [])
        if x and y:
            ax.plot(x, y, color='red', alpha=0.5, linewidth=1.5)
            ax.scatter([x[0]], [y[0]], color='red', marker='o', s=30, zorder=5)
            ax.scatter([x[-1]], [y[-1]], color='red', marker='x', s=40, zorder=5)
            ax.annotate(str(t["id"]), (x[0], y[0]), fontsize=6, color='red')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_xlabel("X (0=left, 1=right)")
    ax.set_ylabel("Y (0=top, 1=bottom)")
    ax.set_title("All Paths (o=start, x=end)")
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    in_p = mpatches.Patch(color='green', alpha=0.5, label=f'IN ({len(in_traj)})')
    out_p = mpatches.Patch(color='red', alpha=0.5, label=f'OUT ({len(out_traj)})')
    ax.legend(handles=[in_p, out_p], fontsize=8)
    
    # Draw best counting line
    best_line = results.get("best_counting_line")
    if best_line:
        ax.axhline(y=best_line["y_line"], color='yellow', linestyle='--', linewidth=2,
                   label=f'Best line y={best_line["y_line"]:.2f} ({best_line["accuracy"]*100:.0f}%)')
        ax.legend(fontsize=7)
    
    # 2. Counting line accuracy sweep
    ax = axes[0, 1]
    sweep = results.get("crossing_line_sweep", [])
    if sweep:
        y_lines = [r["y_line"] for r in sweep]
        accuracies = [r["accuracy"] * 100 for r in sweep]
        correct_in = [r["correct_in"] for r in sweep]
        correct_out = [r["correct_out"] for r in sweep]
        
        ax.plot(y_lines, accuracies, 'b-', linewidth=2, label='Total accuracy')
        ax.fill_between(y_lines, accuracies, alpha=0.1, color='blue')
        
        ax2 = ax.twinx()
        ax2.plot(y_lines, correct_in, 'g--', linewidth=1, alpha=0.7, label='Correct IN')
        ax2.plot(y_lines, correct_out, 'r--', linewidth=1, alpha=0.7, label='Correct OUT')
        
        if best_line:
            ax.axvline(x=best_line["y_line"], color='yellow', linestyle='--',
                       label=f'Best: y={best_line["y_line"]:.2f}')
        
        ax.set_xlabel("Counting Line Y Position")
        ax.set_ylabel("Accuracy %", color='blue')
        ax2.set_ylabel("Correct Counts", color='gray')
        ax.set_title("Counting Line Position vs Accuracy")
        ax.legend(loc='lower left', fontsize=7)
        ax2.legend(loc='lower right', fontsize=7)
    
    # 3. Direction angles (polar plot)
    ax = axes[0, 2]
    pm = results.get("path_metrics", {})
    in_angles = pm.get("in_angles", [])
    out_angles = pm.get("out_angles", [])
    if in_angles or out_angles:
        import math
        # Remove the rectangular axes and add a polar one
        pos = ax.get_position()
        ax.remove()
        ax = fig.add_axes(pos, polar=True)
        
        if in_angles:
            angles_rad = [math.radians(a) for a in in_angles]
            ax.scatter(angles_rad, [1]*len(angles_rad), color='green', s=60, alpha=0.6, label='IN')
        if out_angles:
            angles_rad = [math.radians(a) for a in out_angles]
            ax.scatter(angles_rad, [1]*len(angles_rad), color='red', s=60, alpha=0.6, label='OUT')
        ax.set_title("Movement Direction\n(0°=right, 90°=down)", pad=20)
        ax.legend(fontsize=7, loc='upper right')
    
    # 4. Path linearity distribution
    ax = axes[1, 0]
    in_lin = pm.get("in_linearities", [])
    out_lin = pm.get("out_linearities", [])
    if in_lin or out_lin:
        bins = [i * 0.05 for i in range(21)]
        if in_lin:
            ax.hist(in_lin, bins=bins, alpha=0.6, color='green', label=f'IN (avg={sum(in_lin)/len(in_lin):.2f})')
        if out_lin:
            ax.hist(out_lin, bins=bins, alpha=0.6, color='red', label=f'OUT (avg={sum(out_lin)/len(out_lin):.2f})')
        ax.axvline(x=0.8, color='yellow', linestyle='--', alpha=0.7, label='Straight threshold')
        ax.set_xlabel("Linearity (1.0 = perfectly straight)")
        ax.set_ylabel("Count")
        ax.set_title("Path Linearity\n(>0.8 = line crossing works, <0.8 = need trajectory analysis)")
        ax.legend(fontsize=7)
    
    # 5. Speed distribution
    ax = axes[1, 1]
    in_speeds = pm.get("in_speeds", [])
    out_speeds = pm.get("out_speeds", [])
    if in_speeds or out_speeds:
        bins = 15
        if in_speeds:
            ax.hist(in_speeds, bins=bins, alpha=0.6, color='green', label=f'IN ({len(in_speeds)})')
        if out_speeds:
            ax.hist(out_speeds, bins=bins, alpha=0.6, color='red', label=f'OUT ({len(out_speeds)})')
        ax.set_xlabel("Avg Speed (normalized units/frame)")
        ax.set_ylabel("Count")
        ax.set_title("Movement Speed Distribution")
        ax.legend(fontsize=7)
    
    # 6. Strategy recommendation text
    ax = axes[1, 2]
    ax.axis('off')
    
    text = "COUNTING STRATEGY ANALYSIS\n"
    text += "=" * 40 + "\n\n"
    
    # Analyze what strategy works best
    all_lin = in_lin + out_lin
    avg_linearity = sum(all_lin) / len(all_lin) if all_lin else 0
    
    if best_line:
        text += f"Best counting line: Y = {best_line['y_line']:.2f}\n"
        text += f"  Accuracy: {best_line['accuracy']*100:.1f}%\n"
        text += f"  Correct IN: {best_line['correct_in']}, OUT: {best_line['correct_out']}\n\n"
    
    if avg_linearity > 0.85:
        text += "PATHS ARE STRAIGHT (linearity > 0.85)\n"
        text += "=> Simple LINE CROSSING is best strategy\n"
        text += "=> Use a single horizontal counting line\n\n"
    elif avg_linearity > 0.6:
        text += "PATHS ARE MODERATE (linearity 0.6-0.85)\n"
        text += "=> TWO-ZONE approach recommended\n"
        text += "=> IN-zone + OUT-zone with direction\n\n"
    else:
        text += "PATHS ARE CURVED (linearity < 0.6)\n"
        text += "=> TRAJECTORY-BASED counting needed\n"
        text += "=> Track full path & classify direction\n\n"
    
    # Direction separation
    if in_angles and out_angles:
        avg_in_angle = sum(in_angles) / len(in_angles)
        avg_out_angle = sum(out_angles) / len(out_angles)
        angle_sep = abs(avg_in_angle - avg_out_angle)
        text += f"IN avg direction: {avg_in_angle:.1f}°\n"
        text += f"OUT avg direction: {avg_out_angle:.1f}°\n"
        text += f"Separation: {angle_sep:.1f}°\n"
        if angle_sep > 90:
            text += "=> GREAT direction separation\n\n"
        elif angle_sep > 45:
            text += "=> OK direction separation\n\n"
        else:
            text += "=> POOR separation - need area/zone cues\n\n"
    
    if in_speeds and out_speeds:
        text += f"IN avg speed: {sum(in_speeds)/len(in_speeds):.4f}\n"
        text += f"OUT avg speed: {sum(out_speeds)/len(out_speeds):.4f}\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    path_plot = output_path.replace("_analysis.png", "_paths.png")
    plt.savefig(path_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Path analysis plots saved to: {path_plot}")


def plot_heatmap_analysis(results, output_path, frame_size=None):
    """Generate heatmap and overlap/fragmentation analysis plots."""
    if not HAS_MATPLOTLIB:
        return
    
    try:
        from matplotlib.colors import LinearSegmentedColormap
        import numpy as np
        HAS_NUMPY = True
    except ImportError:
        HAS_NUMPY = False
    
    hm = results.get("heatmap", {})
    start_pos = hm.get("start_positions", {})
    end_pos = hm.get("end_positions", {})
    fragments = results.get("potential_fragments", [])
    xp = results.get("x_positions", {})
    
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    fig.suptitle("Zone Heatmap & Track Quality Analysis",
                 fontsize=14, fontweight='bold')
    
    # 1. Start position scatter (where tracks first appear)
    ax = axes[0, 0]
    for label, color, marker in [("in", "green", "o"), ("out", "red", "s"), ("unlabeled", "gray", "^")]:
        pts = start_pos.get(label, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, color=color, marker=marker, s=60, alpha=0.7,
                      label=f'{label.upper()} ({len(pts)})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Where Tracks FIRST Appear\n(spawn zones)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    
    # 2. End position scatter (where tracks disappear)
    ax = axes[0, 1]
    for label, color, marker in [("in", "green", "o"), ("out", "red", "s"), ("unlabeled", "gray", "^")]:
        pts = end_pos.get(label, [])
        if pts:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.scatter(xs, ys, color=color, marker=marker, s=60, alpha=0.7,
                      label=f'{label.upper()} ({len(pts)})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.invert_yaxis()
    ax.set_aspect('equal')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Where Tracks DISAPPEAR\n(exit zones)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)
    
    # 3. Density heatmap (all positions combined)
    ax = axes[0, 2]
    if HAS_NUMPY:
        all_pts = []
        for label in ["in", "out"]:
            all_pts.extend(start_pos.get(label, []))
            all_pts.extend(end_pos.get(label, []))
        if all_pts:
            xs = [p[0] for p in all_pts]
            ys = [p[1] for p in all_pts]
            heatmap, xedges, yedges = np.histogram2d(xs, ys, bins=15, range=[[0, 1], [0, 1]])
            ax.imshow(heatmap.T, extent=[0, 1, 1, 0], aspect='equal', cmap='hot', interpolation='gaussian')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title("Activity Density Heatmap\n(where to place counting zones)")
    else:
        ax.text(0.5, 0.5, "numpy required\nfor heatmap", ha='center', va='center', transform=ax.transAxes)
    
    # 4. X-position distribution (horizontal lane analysis)
    ax = axes[1, 0]
    in_x_first = xp.get("in_first", [])
    out_x_first = xp.get("out_first", [])
    in_x_last = xp.get("in_last", [])
    out_x_last = xp.get("out_last", [])
    bins = [i * 0.05 for i in range(21)]
    has_data = False
    if in_x_first:
        ax.hist(in_x_first, bins=bins, alpha=0.4, color='green', label=f'IN start ({len(in_x_first)})')
        has_data = True
    if out_x_first:
        ax.hist(out_x_first, bins=bins, alpha=0.4, color='red', label=f'OUT start ({len(out_x_first)})')
        has_data = True
    if in_x_last:
        ax.hist(in_x_last, bins=bins, alpha=0.3, color='darkgreen', linestyle='--', label=f'IN end')
        has_data = True
    if out_x_last:
        ax.hist(out_x_last, bins=bins, alpha=0.3, color='darkred', linestyle='--', label=f'OUT end')
        has_data = True
    if has_data:
        ax.set_xlabel("X Position (0=left, 1=right)")
        ax.set_ylabel("Count")
        ax.set_title("Horizontal Lane Usage\n(do IN/OUT use different sides?)")
        ax.legend(fontsize=7)
    
    # 5. Track fragmentation / overlap chart
    ax = axes[1, 1]
    if fragments:
        ids_a = [f["track_a"] for f in fragments]
        ids_b = [f["track_b"] for f in fragments]
        gaps = [f["frame_gap"] for f in fragments]
        dists = [f["position_distance"] for f in fragments]
        
        labels_text = [f"{a}->{b}" for a, b in zip(ids_a, ids_b)]
        x_pos = range(len(fragments))
        
        ax.bar(x_pos, gaps, color='orange', alpha=0.7, label='Frame gap')
        ax2 = ax.twinx()
        ax2.plot(x_pos, dists, 'ro-', markersize=5, label='Position dist')
        
        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(labels_text, rotation=45, fontsize=7)
        ax.set_xlabel("Track Pair (A -> B)")
        ax.set_ylabel("Frame Gap", color='orange')
        ax2.set_ylabel("Position Distance", color='red')
        ax.set_title(f"Potential Track Fragments ({len(fragments)} pairs)\n(same person, multiple IDs?)")
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
    else:
        ax.text(0.5, 0.5, "No fragmented tracks detected\n(good tracking quality!)",
               ha='center', va='center', transform=ax.transAxes, fontsize=12, color='green')
        ax.set_title("Track Fragmentation Check")
    
    # 6. Summary text
    ax = axes[1, 2]
    ax.axis('off')
    
    text = "ZONE & QUALITY SUMMARY\n"
    text += "=" * 40 + "\n\n"
    
    # X-position insight
    if in_x_first and out_x_first:
        in_x_avg = sum(in_x_first) / len(in_x_first)
        out_x_avg = sum(out_x_first) / len(out_x_first)
        x_sep = abs(in_x_avg - out_x_avg)
        text += f"IN avg X start:  {in_x_avg:.3f}\n"
        text += f"OUT avg X start: {out_x_avg:.3f}\n"
        text += f"X separation:    {x_sep:.3f}\n"
        if x_sep > 0.2:
            text += "=> IN/OUT use DIFFERENT lanes!\n"
            text += "   Consider ROI mask or X-based zones\n\n"
        else:
            text += "=> IN/OUT use SAME lane\n"
            text += "   Y-based counting is sufficient\n\n"
    
    # Fragment insight
    text += f"Potential fragments: {len(fragments)}\n"
    if fragments:
        text += "WARNING: Some tracks may be the same person\n"
        text += "tracked with different IDs. Check:\n"
        for f in fragments[:5]:
            text += f"  ID {f['track_a']} -> {f['track_b']} "
            text += f"(gap={f['frame_gap']}f, dist={f['position_distance']:.3f})\n"
        if len(fragments) > 5:
            text += f"  ... +{len(fragments) - 5} more\n"
        text += "\nThis can cause double-counting!\n"
    else:
        text += "No fragments detected - clean tracking.\n"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    heatmap_plot = output_path.replace("_analysis.png", "_heatmap.png")
    plt.savefig(heatmap_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap analysis saved to: {heatmap_plot}")


def plot_analysis(results, recommendations, output_path, fps=30.0):
    """Generate comprehensive visual analysis."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle("Ground Truth Analysis - People Counter Threshold Optimization", 
                 fontsize=14, fontweight='bold')
    
    # 1. Confidence Distribution
    ax = axes[0, 0]
    if "confidence" in results:
        in_c = results["confidence"]["in_max_confs"]
        out_c = results["confidence"]["out_max_confs"]
        unl_c = results["confidence"]["unlabeled_max_confs"]
        
        bins = [i * 0.05 for i in range(21)]
        if in_c:
            ax.hist(in_c, bins=bins, alpha=0.6, color='green', label=f'IN ({len(in_c)})')
        if out_c:
            ax.hist(out_c, bins=bins, alpha=0.6, color='red', label=f'OUT ({len(out_c)})')
        if unl_c:
            ax.hist(unl_c, bins=bins, alpha=0.3, color='gray', label=f'Unlabeled ({len(unl_c)})')
        
        if "DETECTION_CONFIDENCE" in recommendations:
            rec_val = recommendations["DETECTION_CONFIDENCE"]["value"]
            ax.axvline(x=rec_val, color='orange', linestyle='--', linewidth=2,
                      label=f'Recommended: {rec_val:.3f}')
        
        ax.set_xlabel("Max Confidence")
        ax.set_ylabel("Count")
        ax.set_title("Confidence Distribution (max per track)")
        ax.legend(fontsize=8)
    
    # 2. Confidence Sweep (Recall vs Threshold)
    ax = axes[0, 1]
    if "confidence_sweep" in results:
        sweep = results["confidence_sweep"]
        thresholds = sorted(sweep.keys())
        recalls = [sweep[t]["recall"] for t in thresholds]
        false_pos = [sweep[t]["false_positives"] for t in thresholds]
        
        ax2 = ax.twinx()
        ax.plot(thresholds, recalls, 'b-o', markersize=4, label='Recall')
        ax2.plot(thresholds, false_pos, 'r--s', markersize=4, label='False Positives')
        
        # Mark 85% recall
        ax.axhline(y=0.85, color='green', linestyle=':', alpha=0.7, label='85% target')
        
        ax.set_xlabel("Confidence Threshold")
        ax.set_ylabel("Recall (labeled tracks caught)", color='blue')
        ax2.set_ylabel("Unlabeled tracks (potential FP)", color='red')
        ax.set_title("Recall vs Confidence Threshold")
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    
    # 3. Area Ratio Distribution
    ax = axes[1, 0]
    if "area" in results:
        in_r = results["area"]["in_ratios"]
        out_r = results["area"]["out_ratios"]
        
        if in_r or out_r:
            all_ratios = in_r + out_r
            bins = 20
            if in_r:
                ax.hist(in_r, bins=bins, alpha=0.6, color='green', label=f'IN ({len(in_r)})')
            if out_r:
                ax.hist(out_r, bins=bins, alpha=0.6, color='red', label=f'OUT ({len(out_r)})')
            
            ax.axvline(x=1.0, color='white', linestyle='-', alpha=0.5, label='No change')
            ax.set_xlabel("Area Ratio (last/first)")
            ax.set_ylabel("Count")
            ax.set_title("Bounding Box Area Change\n(>1 = grew = approaching, <1 = shrank = departing)")
            ax.legend(fontsize=8)
    
    # 4. Y-Position Spawn Zones
    ax = axes[1, 1]
    yp = results.get("y_positions", {})
    if any(yp.values()):
        categories = []
        y_vals = []
        colors = []
        
        for y_val in yp.get("in_first", []):
            categories.append("IN first")
            y_vals.append(y_val)
            colors.append('lightgreen')
        for y_val in yp.get("in_last", []):
            categories.append("IN last")
            y_vals.append(y_val)
            colors.append('darkgreen')
        for y_val in yp.get("out_first", []):
            categories.append("OUT first")
            y_vals.append(y_val)
            colors.append('lightsalmon')
        for y_val in yp.get("out_last", []):
            categories.append("OUT last")
            y_vals.append(y_val)
            colors.append('darkred')
        
        # Scatter plot
        cat_map = {"IN first": 0, "IN last": 1, "OUT first": 2, "OUT last": 3}
        x_vals = [cat_map[c] + (hash(str(y)) % 100) * 0.005 for c, y in zip(categories, y_vals)]
        
        ax.scatter(x_vals, y_vals, c=colors, alpha=0.6, s=30)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["IN\nfirst seen", "IN\nlast seen", "OUT\nfirst seen", "OUT\nlast seen"])
        ax.set_ylabel("Y Position (0=top, 1=bottom)")
        ax.set_title("Spawn & Exit Zones (Y-position)")
        ax.invert_yaxis()
        
        # Draw current thresholds
        ax.axhline(y=0.584, color='yellow', linestyle='--', alpha=0.7, label='Current threshold')
        ax.axhline(y=0.382, color='cyan', linestyle=':', alpha=0.7, label='Current spawn_far')
        ax.axhline(y=0.57, color='magenta', linestyle=':', alpha=0.7, label='Current spawn_near')
        ax.legend(fontsize=7)
    
    # 5. Movement Trajectories (Y over time) with counting line
    ax = axes[2, 0]
    traj = results.get("trajectories", {})
    for t in traj.get("in", [])[:15]:  # Limit to 15 for readability
        frames = list(range(len(t["y"])))
        ax.plot(frames, t["y"], color='green', alpha=0.4, linewidth=1)
    for t in traj.get("out", [])[:15]:
        frames = list(range(len(t["y"])))
        ax.plot(frames, t["y"], color='red', alpha=0.4, linewidth=1)
    
    # Best counting line
    best_line = results.get("best_counting_line")
    if best_line:
        ax.axhline(y=best_line["y_line"], color='yellow', linestyle='--', linewidth=2,
                   label=f'Best line y={best_line["y_line"]:.2f} ({best_line["accuracy"]*100:.0f}%)')
    
    ax.set_xlabel("Frames since first detection")
    ax.set_ylabel("Y Position (0=top, 1=bottom)")
    ax.set_title("Movement Trajectories (Y over time)")
    ax.invert_yaxis()
    in_patch = mpatches.Patch(color='green', alpha=0.4, label='IN')
    out_patch = mpatches.Patch(color='red', alpha=0.4, label='OUT')
    handles = [in_patch, out_patch]
    if best_line:
        from matplotlib.lines import Line2D
        handles.append(Line2D([0], [0], color='yellow', linestyle='--', label=f'Best line'))
    ax.legend(handles=handles, fontsize=8)
    
    # 6. Recommendations Summary
    ax = axes[2, 1]
    ax.axis('off')
    
    text = "RECOMMENDED THRESHOLDS\n"
    text += "=" * 40 + "\n\n"
    
    for param, info in recommendations.items():
        text += f"{param}:\n"
        text += f"  Value: {info['value']}\n"
        text += f"  {info['reasoning']}\n\n"
    
    text += "\nCopy these to Counter_Cam.py QUICK CONFIG"
    
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=9,
           verticalalignment='top', fontfamily='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Analysis plots saved to: {output_path}")


def print_text_report(results, recommendations, fps=30.0):
    """Print text-only analysis report."""
    print("\n" + "=" * 60)
    print("GROUND TRUTH ANALYSIS REPORT")
    print("=" * 60)
    
    counts = results["counts"]
    print(f"\nTracks: {counts['total']} total | "
          f"{counts['in']} IN | {counts['out']} OUT | "
          f"{counts['unlabeled']} unlabeled")
    
    # Confidence sweep
    if "confidence_sweep" in results:
        print(f"\n{'Threshold':>10} {'Recall':>8} {'Caught':>8} {'Missed':>8} {'FP':>5}")
        print("-" * 45)
        for thresh, info in sorted(results["confidence_sweep"].items()):
            marker = " <-- 85%" if abs(info["recall"] - 0.85) < 0.05 else ""
            print(f"{thresh:>10.2f} {info['recall']:>8.3f} {info['caught']:>8} "
                  f"{info['missed']:>8} {info['false_positives']:>5}{marker}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("RECOMMENDED THRESHOLDS FOR Counter_Cam.py")
    print(f"{'='*60}\n")
    
    for param, info in recommendations.items():
        print(f"  {param} = {info['value']}")
        print(f"    # {info['reasoning']}")
    
    # Path analysis summary
    pm = results.get("path_metrics", {})
    best_line = results.get("best_counting_line")
    
    print(f"\n{'='*60}")
    print("PATH & TRAJECTORY ANALYSIS")
    print(f"{'='*60}")
    
    if best_line:
        print(f"\n  Best counting line: Y = {best_line['y_line']:.2f}")
        print(f"  Accuracy: {best_line['accuracy']*100:.1f}%")
        print(f"  Correct: IN={best_line['correct_in']} OUT={best_line['correct_out']}")
        print(f"  Wrong:   IN={best_line['wrong_in']} OUT={best_line['wrong_out']}")
    
    in_lin = pm.get("in_linearities", [])
    out_lin = pm.get("out_linearities", [])
    all_lin = in_lin + out_lin
    if all_lin:
        avg_lin = sum(all_lin) / len(all_lin)
        print(f"\n  Path linearity: {avg_lin:.3f} (1.0 = straight)")
        if avg_lin > 0.85:
            print("  => Paths are STRAIGHT - simple line crossing will work well")
        elif avg_lin > 0.6:
            print("  => Paths are MODERATE - two-zone approach recommended")
        else:
            print("  => Paths are CURVED - trajectory-based counting needed")
    
    in_angles = pm.get("in_angles", [])
    out_angles = pm.get("out_angles", [])
    if in_angles and out_angles:
        avg_in = sum(in_angles) / len(in_angles)
        avg_out = sum(out_angles) / len(out_angles)
        print(f"\n  IN direction:  {avg_in:.1f}\u00b0")
        print(f"  OUT direction: {avg_out:.1f}\u00b0")
        print(f"  Separation:    {abs(avg_in - avg_out):.1f}\u00b0")
    
    in_speeds = pm.get("in_speeds", [])
    out_speeds = pm.get("out_speeds", [])
    if in_speeds and out_speeds:
        print(f"\n  IN avg speed:  {sum(in_speeds)/len(in_speeds):.5f}")
        print(f"  OUT avg speed: {sum(out_speeds)/len(out_speeds):.5f}")
    
    # X-position
    xp = results.get("x_positions", {})
    in_x = xp.get("in_first", [])
    out_x = xp.get("out_first", [])
    if in_x and out_x:
        in_x_avg = sum(in_x) / len(in_x)
        out_x_avg = sum(out_x) / len(out_x)
        print(f"\n  IN avg X start:  {in_x_avg:.3f}")
        print(f"  OUT avg X start: {out_x_avg:.3f}")
        x_sep = abs(in_x_avg - out_x_avg)
        if x_sep > 0.2:
            print(f"  => Different horizontal lanes (sep={x_sep:.3f}) - consider X-based zones")
        else:
            print(f"  => Same horizontal lane (sep={x_sep:.3f}) - Y-based counting ok")
    
    # Track fragments
    fragments = results.get("potential_fragments", [])
    if fragments:
        print(f"\n  WARNING: {len(fragments)} potential track fragments (same person, multiple IDs):")
        for f in fragments[:8]:
            print(f"    ID {f['track_a']} -> {f['track_b']}  "
                  f"(gap={f['frame_gap']}f, dist={f['position_distance']:.3f}, "
                  f"labels: {f['label_a']}/{f['label_b']})")
        if len(fragments) > 8:
            print(f"    ... +{len(fragments) - 8} more")
    else:
        print(f"\n  Track quality: No fragments detected (clean tracking)")
    
    print(f"\n{'='*60}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python gt_analyzer.py <ground_truth_file.json> [more_files...]")
        print("\nAnalyzes ground truth data and generates threshold recommendations.")
        sys.exit(1)
    
    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
        
        print(f"\nAnalyzing: {filepath}")
        data = load_ground_truth(filepath)
        
        fps = data.get("metadata", {}).get("fps", 30.0)
        
        results = analyze_tracks(data)
        recommendations = compute_recommendations(results, fps)
        
        # Text report always
        print_text_report(results, recommendations, fps)
        
        # Visual report if matplotlib available
        if HAS_MATPLOTLIB:
            plot_path = filepath.replace(".json", "_analysis.png")
            plot_analysis(results, recommendations, plot_path, fps)
            
            # Path-specific analysis
            frame_size = data.get("metadata", {}).get("frame_size")
            plot_path_analysis(results, recommendations, plot_path, frame_size, fps)
            
            # Heatmap and overlap analysis
            plot_heatmap_analysis(results, plot_path, frame_size)
        
        # Save recommendations as separate JSON
        rec_path = filepath.replace(".json", "_recommendations.json")
        with open(rec_path, 'w') as f:
            json.dump({
                "quick_config_values": {k: v["value"] for k, v in recommendations.items()},
                "details": recommendations,
            }, f, indent=2)
        print(f"Recommendations saved to: {rec_path}")


if __name__ == "__main__":
    main()
