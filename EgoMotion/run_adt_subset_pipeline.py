import argparse
import csv
import io
import json
import os
import time
import urllib.request
import zipfile
from typing import Dict, List, Tuple

import numpy as np


def quat_xyzw_to_rot(q):
	x, y, z, w = q
	xx, yy, zz = x * x, y * y, z * z
	xy, xz, yz = x * y, x * z, y * z
	wx, wy, wz = w * x, w * y, w * z
	return np.array(
		[
			[1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
			[2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
			[2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
		],
		dtype=np.float64,
	)


def safe_diff(x, t):
	dt = np.diff(t)
	dt[dt == 0] = 1e-6
	dx = np.diff(x, axis=0)
	shape = (len(dt),) + (1,) * (dx.ndim - 1)
	return dx / dt.reshape(shape)


def interp_cols(t_src, arr_src, t_dst):
	out = np.zeros((len(t_dst), arr_src.shape[1]), dtype=np.float64)
	for i in range(arr_src.shape[1]):
		out[:, i] = np.interp(t_dst, t_src, arr_src[:, i])
	return out


def find_rgb_for_gt(gt_zip_path):
	base = os.path.basename(gt_zip_path)
	# ADT_xxx_main_groundtruth.zip -> ADT_xxx_preview_rgb.mp4
	rgb_name = base.replace("_main_groundtruth.zip", "_preview_rgb.mp4")
	local_candidate = os.path.join("adt_probe", "rgb_samples", rgb_name)
	if os.path.exists(local_candidate):
		return local_candidate
	return rgb_name


def load_gt(gt_zip):
	with zipfile.ZipFile(gt_zip, "r") as zf:
		names = set(zf.namelist())
		required = {
			"Skeleton_T.json",
			"aria_trajectory.csv",
			"scene_objects.csv",
			"3d_bounding_box.csv",
			"instances.json",
		}
		miss = sorted(required - names)
		if miss:
			raise ValueError("Missing required files in zip: " + ", ".join(miss))

		sk = json.loads(zf.read("Skeleton_T.json").decode("utf-8"))
		traj_txt = zf.read("aria_trajectory.csv").decode("utf-8", errors="ignore")
		scene_txt = zf.read("scene_objects.csv").decode("utf-8", errors="ignore")
		bbox_txt = zf.read("3d_bounding_box.csv").decode("utf-8", errors="ignore")
		instances = json.loads(zf.read("instances.json").decode("utf-8"))

	return sk, traj_txt, scene_txt, bbox_txt, instances


def parse_trunk(traj_txt):
	rows = list(csv.DictReader(io.StringIO(traj_txt)))
	t = np.array([float(r["tracking_timestamp_us"]) * 1e-6 for r in rows], dtype=np.float64)
	q = np.array(
		[
			[
				float(r["qx_world_device"]),
				float(r["qy_world_device"]),
				float(r["qz_world_device"]),
				float(r["qw_world_device"]),
			]
			for r in rows
		],
		dtype=np.float64,
	)
	v_dev = np.array(
		[
			[
				float(r["device_linear_velocity_x_device"]),
				float(r["device_linear_velocity_y_device"]),
				float(r["device_linear_velocity_z_device"]),
			]
			for r in rows
		],
		dtype=np.float64,
	)
	w_dev = np.array(
		[
			[
				float(r["angular_velocity_x_device"]),
				float(r["angular_velocity_y_device"]),
				float(r["angular_velocity_z_device"]),
			]
			for r in rows
		],
		dtype=np.float64,
	)

	pos = np.array(
		[
			[
				float(r["tx_world_device"]),
				float(r["ty_world_device"]),
				float(r["tz_world_device"]),
			]
			for r in rows
		],
		dtype=np.float64,
	)

	v_world = np.zeros_like(v_dev)
	w_world = np.zeros_like(w_dev)
	for i in range(len(rows)):
		rmat = quat_xyzw_to_rot(q[i])
		v_world[i] = rmat @ v_dev[i]
		w_world[i] = rmat @ w_dev[i]

	trunk6 = np.concatenate([v_world, w_world], axis=1)
	return t, pos, q, trunk6


def parse_skeleton(sk):
	frames = sk["frames"]
	t = np.array([int(f["timestamp_ns"]) * 1e-9 for f in frames], dtype=np.float64)
	joints = np.array([f["joints"] for f in frames], dtype=np.float64)  # [T,51,3]
	vel = np.zeros_like(joints)
	vel[1:] = safe_diff(joints, t)
	vel = vel.reshape(len(t), -1)  # [T,153]
	return t, joints, vel


def parse_objects(scene_txt, bbox_txt, instances):
	scene_rows = list(csv.DictReader(io.StringIO(scene_txt)))
	bbox_rows = list(csv.DictReader(io.StringIO(bbox_txt)))

	# object center/orientation in world
	pose = {}
	for r in scene_rows:
		uid = int(r["object_uid"])
		pose[uid] = {
			"t": np.array([float(r["t_wo_x[m]"]), float(r["t_wo_y[m]"]), float(r["t_wo_z[m]"])], dtype=np.float64),
			"q": np.array([float(r["q_wo_x"]), float(r["q_wo_y"]), float(r["q_wo_z"]), float(r["q_wo_w"])], dtype=np.float64),
		}

	# local bbox extents
	ext = {}
	for r in bbox_rows:
		uid = int(r["object_uid"])
		ext[uid] = {
			"xmin": float(r["p_local_obj_xmin[m]"]),
			"xmax": float(r["p_local_obj_xmax[m]"]),
			"ymin": float(r["p_local_obj_ymin[m]"]),
			"ymax": float(r["p_local_obj_ymax[m]"]),
			"zmin": float(r["p_local_obj_zmin[m]"]),
			"zmax": float(r["p_local_obj_zmax[m]"]),
		}

	objects = []
	for uid, p in pose.items():
		if uid not in ext:
			continue
		inst = instances.get(str(uid), {})
		if inst.get("instance_type") == "human":
			continue
		e = ext[uid]
		center_local = np.array(
			[0.5 * (e["xmin"] + e["xmax"]), 0.5 * (e["ymin"] + e["ymax"]), 0.5 * (e["zmin"] + e["zmax"])],
			dtype=np.float64,
		)
		size = np.array([e["xmax"] - e["xmin"], e["ymax"] - e["ymin"], e["zmax"] - e["zmin"]], dtype=np.float64)

		r_wo = quat_xyzw_to_rot(p["q"])
		center_world = p["t"] + r_wo @ center_local

		objects.append(
			{
				"uid": uid,
				"center_world": center_world,
				"size": np.maximum(size, 1e-4),
				"category": inst.get("category", "unknown"),
			}
		)

	return objects


def build_occupancy_3d(
	trunk_t,
	trunk_pos,
	trunk_q,
	skel_t,
	objects,
	xlim=(-1.0, 5.0),
	ylim=(-3.0, 3.0),
	zlim=(0.0, 2.5),
	voxel=0.2,
):
	nx = int(np.ceil((xlim[1] - xlim[0]) / voxel))
	ny = int(np.ceil((ylim[1] - ylim[0]) / voxel))
	nz = int(np.ceil((zlim[1] - zlim[0]) / voxel))

	# Interpolate trunk pose to skeleton timestamps
	pos_i = interp_cols(trunk_t, trunk_pos, skel_t)
	q_i = np.zeros((len(skel_t), 4), dtype=np.float64)
	for j in range(4):
		q_i[:, j] = np.interp(skel_t, trunk_t, trunk_q[:, j])
	q_i = q_i / np.maximum(np.linalg.norm(q_i, axis=1, keepdims=True), 1e-8)

	occ = np.zeros((len(skel_t), nz, ny, nx), dtype=np.uint8)

	for t_idx in range(len(skel_t)):
		p = pos_i[t_idx]
		r_wd = quat_xyzw_to_rot(q_i[t_idx])
		r_dw = r_wd.T

		for obj in objects:
			c_world = obj["center_world"]
			half = 0.5 * obj["size"]

			c_local = r_dw @ (c_world - p)

			mn = c_local - half
			mx = c_local + half

			ix0 = int(np.floor((mn[0] - xlim[0]) / voxel))
			ix1 = int(np.ceil((mx[0] - xlim[0]) / voxel))
			iy0 = int(np.floor((mn[1] - ylim[0]) / voxel))
			iy1 = int(np.ceil((mx[1] - ylim[0]) / voxel))
			iz0 = int(np.floor((mn[2] - zlim[0]) / voxel))
			iz1 = int(np.ceil((mx[2] - zlim[0]) / voxel))

			ix0, ix1 = max(0, ix0), min(nx, ix1)
			iy0, iy1 = max(0, iy0), min(ny, iy1)
			iz0, iz1 = max(0, iz0), min(nz, iz1)

			if ix0 < ix1 and iy0 < iy1 and iz0 < iz1:
				occ[t_idx, iz0:iz1, iy0:iy1, ix0:ix1] = 1

	grid_meta = {
		"xlim": list(xlim),
		"ylim": list(ylim),
		"zlim": list(zlim),
		"voxel": float(voxel),
		"shape": [int(nz), int(ny), int(nx)],
	}
	return occ, grid_meta


def _safe_mkdir(path: str) -> None:
	os.makedirs(path, exist_ok=True)


def _download_with_retry(url: str, dst: str, expected_size: int, retries: int = 3) -> str:
	if os.path.exists(dst) and os.path.getsize(dst) == expected_size:
		return "skip"

	tmp = dst + ".part"
	if os.path.exists(tmp):
		try:
			os.remove(tmp)
		except OSError:
			pass

	last_err = None
	for i in range(retries):
		try:
			urllib.request.urlretrieve(url, tmp)
			if os.path.getsize(tmp) != expected_size:
				raise RuntimeError(
					f"size mismatch for {os.path.basename(dst)}: "
					f"got {os.path.getsize(tmp)}, expected {expected_size}"
				)
			os.replace(tmp, dst)
			return "ok"
		except Exception as err:
			last_err = err
			if os.path.exists(tmp):
				try:
					os.remove(tmp)
				except OSError:
					pass
			time.sleep(1.5 * (i + 1))

	raise last_err


def _select_sequences(
	urls_json_path: str,
	categories: List[str],
	max_sequences: int,
) -> List[Tuple[str, Dict]]:
	with open(urls_json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	seqs = data.get("sequences", {})
	selected = []
	required = {"video_main_rgb", "main_groundtruth"}

	for seq_name, items in sorted(seqs.items()):
		if not required.issubset(items.keys()):
			continue

		if categories:
			keep = False
			for cat in categories:
				if f"release_{cat}_seq" in seq_name:
					keep = True
					break
			if not keep:
				continue

		selected.append((seq_name, items))
		if max_sequences > 0 and len(selected) >= max_sequences:
			break

	return selected


def _download_subset(
	selected: List[Tuple[str, Dict]],
	download_root: str,
) -> List[Dict[str, str]]:
	records = []
	ok = 0
	skip = 0
	fail = 0

	for idx, (seq_name, items) in enumerate(selected, 1):
		seq_dir = os.path.join(download_root, seq_name)
		_safe_mkdir(seq_dir)

		seq_rec = {"sequence": seq_name}

		for key in ("video_main_rgb", "main_groundtruth"):
			meta = items[key]
			dst = os.path.join(seq_dir, meta["filename"])
			try:
				status = _download_with_retry(
					url=meta["download_url"],
					dst=dst,
					expected_size=int(meta["file_size_bytes"]),
				)
				if status == "ok":
					ok += 1
				else:
					skip += 1
				seq_rec[key] = dst
			except Exception as err:
				fail += 1
				seq_rec[key] = ""
				seq_rec[f"{key}_error"] = str(err)

		records.append(seq_rec)
		print(f"download {idx}/{len(selected)} | ok={ok} skip={skip} fail={fail}")

	print(f"download summary | ok={ok} skip={skip} fail={fail}")
	return records


def _process_one(gt_zip: str, rgb_video: str, out_dir: str, voxel: float) -> Dict:
	sk, traj_txt, scene_txt, bbox_txt, instances = load_gt(gt_zip)
	trunk_t, trunk_pos, trunk_q, trunk6 = parse_trunk(traj_txt)
	skel_t, skel_xyz, skel_vel153 = parse_skeleton(sk)
	objects = parse_objects(scene_txt, bbox_txt, instances)

	trunk6_i = interp_cols(trunk_t, trunk6, skel_t).astype(np.float32)
	occ, grid_meta = build_occupancy_3d(
		trunk_t=trunk_t,
		trunk_pos=trunk_pos,
		trunk_q=trunk_q,
		skel_t=skel_t,
		objects=objects,
		voxel=voxel,
	)

	_safe_mkdir(out_dir)
	base = os.path.splitext(os.path.basename(gt_zip))[0]
	npz_path = os.path.join(out_dir, f"{base}_three_decoder_data.npz")
	meta_path = os.path.join(out_dir, f"{base}_three_decoder_data_meta.json")

	np.savez_compressed(
		npz_path,
		t_sec=skel_t.astype(np.float64),
		trunk6=trunk6_i,
		skeleton_xyz=skel_xyz.astype(np.float32),
		skeleton_vel153=skel_vel153.astype(np.float32),
		occupancy3d=occ,
	)

	meta = {
		"gt_zip": gt_zip,
		"rgb_video": rgb_video,
		"num_frames": int(len(skel_t)),
		"trunk6_shape": list(trunk6_i.shape),
		"skeleton_vel153_shape": list(skel_vel153.shape),
		"occupancy3d_shape": list(occ.shape),
		"num_objects_used": int(len(objects)),
		"grid": grid_meta,
	}
	with open(meta_path, "w", encoding="utf-8") as f:
		json.dump(meta, f, indent=2)

	return {
		"npz": npz_path,
		"meta": meta_path,
		"frames": int(len(skel_t)),
		"occ_shape": list(occ.shape),
	}


def main() -> None:
	parser = argparse.ArgumentParser(
		description="Download ADT subset and prepare three-decoder training data"
	)
	parser.add_argument(
		"--urls-json",
		type=str,
		default="ADT_download_urls.json",
		help="Path to ADT_download_urls.json",
	)
	parser.add_argument(
		"--categories",
		type=str,
		default="work_skeleton,meal_skeleton,decoration_skeleton",
		help="Comma-separated categories (use empty string for all)",
	)
	parser.add_argument(
		"--max-sequences",
		type=int,
		default=3,
		help="Max number of sequences to process, -1 for all selected",
	)
	parser.add_argument(
		"--download-root",
		type=str,
		default="adt_pipeline_data/raw",
		help="Folder for downloaded files",
	)
	parser.add_argument(
		"--processed-root",
		type=str,
		default="adt_pipeline_data/processed",
		help="Folder for processed npz labels",
	)
	parser.add_argument(
		"--voxel",
		type=float,
		default=0.4,
		help="3D occupancy voxel size in meters",
	)
	parser.add_argument(
		"--skip-download",
		action="store_true",
		help="Skip downloading and only process existing files under download-root",
	)
	args = parser.parse_args()

	categories = [c.strip() for c in args.categories.split(",") if c.strip()]
	max_seq = args.max_sequences if args.max_sequences > 0 else 10**9

	selected = _select_sequences(
		urls_json_path=args.urls_json,
		categories=categories,
		max_sequences=max_seq,
	)
	if not selected:
		raise RuntimeError("No sequences selected. Check categories or urls json.")

	print(f"selected {len(selected)} sequences")

	if args.skip_download:
		records = []
		for seq_name, items in selected:
			seq_dir = os.path.join(args.download_root, seq_name)
			gt_path = os.path.join(seq_dir, items["main_groundtruth"]["filename"])
			rgb_path = os.path.join(seq_dir, items["video_main_rgb"]["filename"])
			records.append(
				{
					"sequence": seq_name,
					"main_groundtruth": gt_path,
					"video_main_rgb": rgb_path,
				}
			)
	else:
		_safe_mkdir(args.download_root)
		records = _download_subset(selected, args.download_root)

	_safe_mkdir(args.processed_root)
	manifest = {
		"config": {
			"urls_json": args.urls_json,
			"categories": categories,
			"max_sequences": args.max_sequences,
			"voxel": args.voxel,
		},
		"samples": [],
	}

	done = 0
	failed = 0
	for i, rec in enumerate(records, 1):
		seq = rec.get("sequence", "unknown")
		gt_zip = rec.get("main_groundtruth", "")
		rgb_video = rec.get("video_main_rgb", "")

		if not gt_zip or not os.path.exists(gt_zip):
			print(f"process {i}/{len(records)} skip {seq}: missing gt zip")
			failed += 1
			continue

		if not rgb_video:
			rgb_video = find_rgb_for_gt(gt_zip)

		try:
			out = _process_one(
				gt_zip=gt_zip,
				rgb_video=rgb_video,
				out_dir=args.processed_root,
				voxel=args.voxel,
			)
			manifest["samples"].append(
				{
					"sequence": seq,
					"rgb_video": rgb_video,
					"gt_zip": gt_zip,
					"npz": out["npz"],
					"meta": out["meta"],
					"frames": out["frames"],
					"occ_shape": out["occ_shape"],
				}
			)
			done += 1
			print(f"process {i}/{len(records)} done {seq}")
		except Exception as err:
			failed += 1
			print(f"process {i}/{len(records)} failed {seq}: {err}")

	manifest_path = os.path.join(args.processed_root, "manifest.json")
	with open(manifest_path, "w", encoding="utf-8") as f:
		json.dump(manifest, f, indent=2)

	print(f"finished | done={done} failed={failed}")
	print("manifest:", manifest_path)


if __name__ == "__main__":
	main()
