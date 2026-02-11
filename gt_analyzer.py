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
    
    # === Movement Trajectories ===
    in_trajectories = []
    out_trajectories = []
    for t in in_tracks.values():
        y_hist = t.get("centroid_y_history", [])
        a_hist = t.get("area_history", [])
        if y_hist:
            in_trajectories.append({"y": y_hist, "area": a_hist, "id": t["track_id"]})
    for t in out_tracks.values():
        y_hist = t.get("centroid_y_history", [])
        a_hist = t.get("area_history", [])
        if y_hist:
            out_trajectories.append({"y": y_hist, "area": a_hist, "id": t["track_id"]})
    
    results["trajectories"] = {
        "in": in_trajectories,
        "out": out_trajectories,
    }
    
    # === Track Duration ===
    in_durations = [t["total_frames_visible"] for t in in_tracks.values()]
    out_durations = [t["total_frames_visible"] for t in out_tracks.values()]
    unlabeled_durations = [t["total_frames_visible"] for t in unlabeled.values()]
    
    results["durations"] = {
        "in": in_durations,
        "out": out_durations,
        "unlabeled": unlabeled_durations,
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
    
    # 5. Movement Trajectories (Y over time)
    ax = axes[2, 0]
    traj = results.get("trajectories", {})
    for t in traj.get("in", [])[:15]:  # Limit to 15 for readability
        frames = list(range(len(t["y"])))
        ax.plot(frames, t["y"], color='green', alpha=0.4, linewidth=1)
    for t in traj.get("out", [])[:15]:
        frames = list(range(len(t["y"])))
        ax.plot(frames, t["y"], color='red', alpha=0.4, linewidth=1)
    
    ax.set_xlabel("Frames since first detection")
    ax.set_ylabel("Y Position (0=top, 1=bottom)")
    ax.set_title("Movement Trajectories (Y over time)")
    ax.invert_yaxis()
    in_patch = mpatches.Patch(color='green', alpha=0.4, label='IN')
    out_patch = mpatches.Patch(color='red', alpha=0.4, label='OUT')
    ax.legend(handles=[in_patch, out_patch], fontsize=8)
    
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
