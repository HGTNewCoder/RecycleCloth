"""Preprocess DeepFashion2 annotation files.

This script scans the annotation folders, finds files whose top-level
``source`` field contains ``shop``, and writes the matching six-digit image
ids to a text file.
"""

import json
import os


DEFAULT_OUTPUT_FILE = "shop_image_ids.txt"


def find_deepfashion_root():
	current_dir = os.getcwd()
	script_dir = os.path.dirname(os.path.abspath(__file__))
	candidates = [
		os.path.join(current_dir, "Datasets", "DeepFashion2", "deepfashion2_original_images"),
		os.path.join(script_dir, "Datasets", "DeepFashion2", "deepfashion2_original_images"),
	]

	for candidate in candidates:
		if os.path.isdir(candidate):
			return candidate

	raise FileNotFoundError("Could not find DeepFashion2/deepfashion2_original_images under the current workspace.")


def iter_annotation_files(deepfashion_root):
	split_directories = [
		os.path.join(deepfashion_root, "train", "annos"),
		os.path.join(deepfashion_root, "test", "annos"),
		os.path.join(deepfashion_root, "validation", "annos"),
		os.path.join(deepfashion_root, "val", "annos"),
	]

	for annos_dir in split_directories:
		if not os.path.isdir(annos_dir):
			continue
		for file_name in sorted(os.listdir(annos_dir)):
			if file_name.endswith(".json"):
				yield os.path.join(annos_dir, file_name)


def collect_shop_image_ids(deepfashion_root):
	shop_image_ids = set()

	for json_file in iter_annotation_files(deepfashion_root):
		with open(json_file, "r", encoding="utf-8") as file_handle:
			annotation = json.load(file_handle)

		source = annotation.get("source", "")
		if "shop" in str(source).lower():
			shop_image_ids.add(os.path.splitext(os.path.basename(json_file))[0])

	return sorted(shop_image_ids)


def write_image_ids(image_ids, output_file):
	with open(output_file, "w", encoding="utf-8") as file_handle:
		file_handle.write("\n".join(image_ids) + ("\n" if image_ids else ""))


def remove_shop_images(deepfashion_root, shop_image_ids_file):
	"""Reads shop image ids from shop_image_ids_file and removes corresponding images and annotations from deepfashion2_original_images."""
	if not os.path.exists(shop_image_ids_file):
		print(f"Error: shop image IDs file not found at {shop_image_ids_file}")
		return

	with open(shop_image_ids_file, "r", encoding="utf-8") as file_handle:
		shop_image_ids = {line.strip() for line in file_handle if line.strip()}

	print(f"Loaded {len(shop_image_ids)} shop image IDs to remove.")

	splits = ["train", "test", "validation", "val"]
	removed_images_count = 0
	removed_annos_count = 0
	not_found_count = 0

	for image_id in shop_image_ids:
		found_any = False
		for split in splits:
			image_path = os.path.join(deepfashion_root, split, "image", f"{image_id}.jpg")
			anno_path = os.path.join(deepfashion_root, split, "annos", f"{image_id}.json")

			# Check and remove image
			if os.path.exists(image_path):
				try:
					os.remove(image_path)
					removed_images_count += 1
					found_any = True
				except Exception as e:
					print(f"Failed to remove image {image_path}: {e}")

			# Check and remove annotation
			if os.path.exists(anno_path):
				try:
					os.remove(anno_path)
					removed_annos_count += 1
					found_any = True
				except Exception as e:
					print(f"Failed to remove annotation {anno_path}: {e}")

		if not found_any:
			not_found_count += 1

	print(f"Successfully removed {removed_images_count} images.")
	print(f"Successfully removed {removed_annos_count} json annotation files.")
	if not_found_count > 0:
		print(f"{not_found_count} shop image IDs had neither images nor annotations found in any split (they might have already been removed).")


def main():
	deepfashion_root = find_deepfashion_root()
	script_dir = os.path.dirname(os.path.abspath(__file__))
	shop_image_ids_file = os.path.join(script_dir, DEFAULT_OUTPUT_FILE)

	# Fallback if scripts/shop_image_ids.txt is not found in script_dir but is in cwd
	if not os.path.exists(shop_image_ids_file):
		cwd_file = os.path.join(os.getcwd(), DEFAULT_OUTPUT_FILE)
		if os.path.exists(cwd_file):
			shop_image_ids_file = cwd_file
		else:
			print(f"{shop_image_ids_file} not found. Collecting shop image IDs first...")
			image_ids = collect_shop_image_ids(deepfashion_root)
			write_image_ids(image_ids, shop_image_ids_file)
			print(f"Wrote {len(image_ids)} shop image ids to {shop_image_ids_file}")

	remove_shop_images(deepfashion_root, shop_image_ids_file)


if __name__ == "__main__":
	main()

