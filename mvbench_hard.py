import os
import json
import argparse
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='check pooling results!')
    parser.add_argument('--pool2', required=True, help='path_to_pool2_folder')
    parser.add_argument('--pool4', required=True, help='path_to_pool4_folder')
    parser.add_argument('--pool8', required=True, help='path_to_pool8_folder')
    parser.add_argument('--pool16', required=True, help='path_to_pool16_folder')
    parser.add_argument('--ldp', required=True, help='path_to_ldp_folder')
    parser.add_argument('--output', required=True, help='out_path') # .json
    return parser.parse_args()

def load_json(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error {file_path}: {e}")
        return None

def check_ans(pred, gt):
    flag = False

    pred_list = pred.lower().split(' ')
    pred_option, pred_content = pred_list[1], ' '.join(pred_list[1:])
    gt_list = gt.lower().split(' ')
    gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
    if gt_content[-1] == '.':
        gt_content = gt_content[:-1]
    print(gt_option, pred_option)
    if pred_option.replace('.', '') in gt_option:
        flag = True
    elif gt_option in pred_option:
        flag = True

    return flag

def check_numbers(nums):
    if nums[0] != 1:
        return False
    if nums[3] != 0:
        return False
    for i in range(1, 3):
        if nums[i] < nums[i - 1]:
            return False
    return True


def get_mvbench_hard(folder1, folder2, folder3, folder4,ldp_folder, output):
    files1 = set(os.listdir(folder1))
    files2 = set(os.listdir(folder2))
    files3 = set(os.listdir(folder3))
    files4 = set(os.listdir(folder4))
    common_files = files1 & files2 & files3 & files4
    print(len(common_files))
    assert len(common_files)==4000 # 完整性校验

    mvbench_hard = {}
    right = [0, 0, 0, 0, 0]
    for filename in tqdm(common_files):
        file1_path = os.path.join(folder1, filename)
        file2_path = os.path.join(folder2, filename)
        file3_path = os.path.join(folder3, filename)
        file4_path = os.path.join(folder4, filename)
        ldp_file_path = os.path.join(ldp_folder, filename)

        json1 = load_json(file1_path)
        json2 = load_json(file2_path)
        json3 = load_json(file3_path)
        json4 = load_json(file4_path)
        ldp_json = load_json(ldp_file_path)

        gt_answer = json1['A']
        pred1 = json1['pred']
        pred2 = json2['pred']
        pred3 = json3['pred']
        pred4 = json4['pred']
        ldp_pred = ldp_json['pred']
        
        if not check_numbers([check_ans(pred1, gt_answer),check_ans(pred2, gt_answer),check_ans(pred3, gt_answer),check_ans(pred4, gt_answer)]):
            continue
        else:
            mvbench_hard[filename] = [check_ans(pred1, gt_answer),check_ans(pred2, gt_answer),check_ans(pred3, gt_answer),check_ans(pred4, gt_answer),check_ans(ldp_pred, gt_answer)]
            right[0] += check_ans(pred1, gt_answer)
            right[1] += check_ans(pred2, gt_answer)
            right[2] += check_ans(pred3, gt_answer)
            right[3] += check_ans(pred4, gt_answer)
            right[4] += check_ans(ldp_pred, gt_answer)
    for i in right:
        print(i/len(mvbench_hard))
    with open(output, 'w') as out_file:
        json.dump(mvbench_hard, out_file, ensure_ascii=False, indent=4)
    print(f"Savee to {output}")



if __name__ == '__main__':
    args = parse_args()
    get_mvbench_hard(args.pool2, args.pool4, args.pool8, args.pool16, args.ldp, args.output)