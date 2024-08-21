import json
import os

def main():
    f_path = "../checkpoint/COCO2017_from_Kaggle/coco2017/annotations/person_keypoints_val2017.json"
    with open(f_path, 'r') as fptr:
        j_str = fptr.read()
    print(f"Parsing file: {f_path}")
    j_dict = json.loads(j_str)
    folder, basename = os.path.split(f_path)
    stem, ext = os.path.splitext(basename)
    f_save = os.path.join(folder, f"{stem}_formatted{ext}")
    with open(f_save, "w") as fptr:
        new_str = json.dumps(j_dict, indent=2)
        fptr.write(new_str)
    print(f"format done : {f_save}")

if __name__ == "__main__":
    main()
