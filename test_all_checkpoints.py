#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
import datetime
import glob
from time import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import csv

import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加igformer路径到sys.path (专门用于igformer测试)
import sys
import os
current_dir = os.getcwd()

# 清理可能干扰的路径
sys.path = [p for p in sys.path if '/Igformer/' not in p]

# 按照train2.py的方式添加路径
sys.path.insert(0, 'Igformer/models1')

# 导入igformer模型
from models import IgformerModel

# 添加安全全局变量以支持PyTorch checkpoint加载
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([IgformerModel])

from data.dataset import E2EDataset
from data.pdb_utils import VOCAB, Residue, Peptide, Protein, AgAbComplex
from evaluation.rmsd import compute_rmsd
from evaluation.tm_score import tm_score
from evaluation.lddt import lddt
from evaluation.dockq import dockq

# 可选依赖，如果不存在则设置为None
try:
    from utils.relax import openmm_relax, rosetta_sidechain_packing
    RELAX_AVAILABLE = True
except ImportError:
    print("Warning: utils.relax not available. Relaxation features will be disabled.")
    RELAX_AVAILABLE = False
    openmm_relax = None
    rosetta_sidechain_packing = None

from utils.logger import print_log
from utils.random_seed import setup_seed

# 尝试导入CONTACT_DIST，如果不存在则使用默认值
try:
    from configs import CONTACT_DIST
except ImportError:
    CONTACT_DIST = 6.0  # 默认接触距离阈值
    print("Warning: CONTACT_DIST not found in configs, using default value 6.0")


def to_cplx(ori_cplx, ab_x, ab_s) -> AgAbComplex:
    """将生成的坐标和序列转换为AgAbComplex对象"""
    heavy_chain, light_chain = [], []
    chain = None
    for residue, residue_x in zip(ab_s, ab_x):
        residue = VOCAB.idx_to_symbol(residue)
        if residue == VOCAB.BOA:
            continue
        elif residue == VOCAB.BOH:
            chain = heavy_chain
            continue
        elif residue == VOCAB.BOL:
            chain = light_chain
            continue
        if chain is None:  # still in antigen region
            continue
        coord, atoms = {}, VOCAB.backbone_atoms + VOCAB.get_sidechain_info(residue)

        for atom, x in zip(atoms, residue_x):
            coord[atom] = x
        chain.append(Residue(
            residue, coord, _id=(len(chain), ' ')
        ))
    heavy_chain = Peptide(ori_cplx.heavy_chain, heavy_chain)
    light_chain = Peptide(ori_cplx.light_chain, light_chain)
    for res, ori_res in zip(heavy_chain, ori_cplx.get_heavy_chain()):
        res.id = ori_res.id
    for res, ori_res in zip(light_chain, ori_cplx.get_light_chain()):
        res.id = ori_res.id

    peptides = {
        ori_cplx.heavy_chain: heavy_chain,
        ori_cplx.light_chain: light_chain
    }
    antibody = Protein(ori_cplx.pdb_id, peptides)
    cplx = AgAbComplex(
        ori_cplx.antigen, antibody, ori_cplx.heavy_chain,
        ori_cplx.light_chain, skip_epitope_cal=True,
        skip_validity_check=True
    )
    cplx.cdr_pos = ori_cplx.cdr_pos
    return cplx


def create_save_dir(base_dir, checkpoint_name):
    """在指定目录下创建一个以checkpoint名称命名的子目录"""
    save_dir = os.path.join(base_dir, f"test_results_{checkpoint_name}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def cal_metrics(inputs):
    """计算单个模型的指标"""
    if len(inputs) == 6:
        mod_pdb, ref_pdb, H, L, A, cdr_type = inputs
        sidechain = False
    elif len(inputs) == 7:
        mod_pdb, ref_pdb, H, L, A, cdr_type, sidechain = inputs
    do_refine = False

    # sidechain packing
    if sidechain and RELAX_AVAILABLE and rosetta_sidechain_packing:
        refined_pdb = mod_pdb[:-4] + '_sidechain.pdb'
        mod_pdb = rosetta_sidechain_packing(mod_pdb, refined_pdb)
    elif sidechain:
        print("Warning: Sidechain packing requested but not available")

    # load complex
    if do_refine and RELAX_AVAILABLE and openmm_relax:
        refined_pdb = mod_pdb[:-4] + '_refine.pdb'
        pdb_id = os.path.split(mod_pdb)[-1]
        print(f'{pdb_id} started refining')
        start = time()
        mod_pdb = openmm_relax(mod_pdb, refined_pdb, excluded_chains=A)  # relax clashes
        print(f'{pdb_id} finished openmm relax, elapsed {round(time() - start)} s')
    elif do_refine:
        print("Warning: Refinement requested but not available")
    
    mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A, skip_epitope_cal=True)
    ref_cplx = AgAbComplex.from_pdb(ref_pdb, H, L, A, skip_epitope_cal=False)

    results = {}
    # 保持原始的 cdr_type 格式，不强制转换为列表
    # 这样既兼容 single CDR (str) 也兼容 multi CDR (list)

    # 1. AAR & CAAR
    epitope = ref_cplx.get_epitope()
    is_contact = []
    if cdr_type is None:  # entire antibody
        gt_s = ref_cplx.get_heavy_chain().get_seq() + ref_cplx.get_light_chain().get_seq()
        pred_s = mod_cplx.get_heavy_chain().get_seq() + mod_cplx.get_light_chain().get_seq()
        # contact
        for chain in [ref_cplx.get_heavy_chain(), ref_cplx.get_light_chain()]:
            for ab_residue in chain:
                contact = False
                for ag_residue, _, _ in epitope:
                    dist = ab_residue.dist_to(ag_residue)
                    if dist < CONTACT_DIST:
                        contact = True
                is_contact.append(int(contact))
    else:
        gt_s, pred_s = '', ''
        # 处理单个或多个 CDR
        cdr_list = [cdr_type] if isinstance(cdr_type, str) else cdr_type
        for cdr in cdr_list:
            gt_cdr = ref_cplx.get_cdr(cdr)
            cur_gt_s = gt_cdr.get_seq()
            cur_pred_s = mod_cplx.get_cdr(cdr).get_seq()
            gt_s += cur_gt_s
            pred_s += cur_pred_s
            # contact
            cur_contact = []
            for ab_residue in gt_cdr:
                contact = False
                for ag_residue, _, _ in epitope:
                    dist = ab_residue.dist_to(ag_residue)
                    if dist < CONTACT_DIST:
                        contact = True
                cur_contact.append(int(contact))
            is_contact.extend(cur_contact)

            hit, chit = 0, 0
            for a, b, contact in zip(cur_gt_s, cur_pred_s, cur_contact):
                if a == b:
                    hit += 1
                    if contact == 1:
                        chit += 1
            results[f'AAR {cdr}'] = hit * 1.0 / len(cur_gt_s)
            results[f'CAAR {cdr}'] = chit * 1.0 / (sum(cur_contact) + 1e-10)

    if len(gt_s) != len(pred_s):
        print_log(f'Length conflict: {len(gt_s)} and {len(pred_s)}', level='WARN')
    hit, chit = 0, 0
    for a, b, contact in zip(gt_s, pred_s, is_contact):
        if a == b:
            hit += 1
            if contact == 1:
                chit += 1
    results['AAR'] = hit * 1.0 / len(gt_s)
    results['CAAR'] = chit * 1.0 / (sum(is_contact) + 1e-10)

    # 2. RMSD(CA) w/o align
    gt_x, pred_x = [], []
    for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
        for chain in [c.get_heavy_chain(), c.get_light_chain()]:
            for i in range(len(chain)):
                xl.append(chain.get_ca_pos(i))
    assert len(gt_x) == len(pred_x), f'coordinates length conflict'
    gt_x, pred_x = np.array(gt_x), np.array(pred_x)
    results['RMSD(CA) aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)
    results['RMSD(CA)'] = compute_rmsd(gt_x, pred_x, aligned=True)

    # 3. TMscore
    try:
        results['TMscore'] = tm_score(mod_cplx.antibody, ref_cplx.antibody)
    except Exception as e:
        print_log(f'Error calculating TMscore: {e}', level='WARN')
        results['TMscore'] = 0.0

    # 4. LDDT
    try:
        score, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
        results['LDDT'] = score
    except Exception as e:
        print_log(f'Error calculating LDDT: {e}', level='WARN')
        results['LDDT'] = 0.0

    # 5. DockQ
    try:
        score = dockq(mod_cplx, ref_cplx, cdrh3_only=True) # consistent with HERN
    except Exception as e:
        print_log(f'Error in dockq: {e}, set to 0', level='ERROR')
        score = 0
    results['DockQ'] = score

    # 计算每个 CDR 区域的 RMSD
    cdr_list = [cdr_type] if isinstance(cdr_type, str) else cdr_type
    for cdr in cdr_list:
        gt_cdr_coords, pred_cdr_coords = [], []
        gt_cdr = ref_cplx.get_cdr(cdr)
        pred_cdr = mod_cplx.get_cdr(cdr)
        for i in range(len(gt_cdr)):
            gt_cdr_coords.append(gt_cdr.get_ca_pos(i))
        for i in range(len(pred_cdr)):
            pred_cdr_coords.append(pred_cdr.get_ca_pos(i))

        gt_cdr_coords = np.array(gt_cdr_coords)
        pred_cdr_coords = np.array(pred_cdr_coords)
        results[f'RMSD(CA) CDR{cdr} aligned'] = compute_rmsd(gt_cdr_coords, pred_cdr_coords, aligned=False)
        results[f'RMSD(CA) CDR{cdr}'] = compute_rmsd(gt_cdr_coords, pred_cdr_coords, aligned=True)

    # 收集每个 CDR 区域的序列
    sequences = {}
    for cdr in cdr_list:
        sequences[f'cdr_{cdr}_gt'] = ref_cplx.get_cdr(cdr).get_seq()
        sequences[f'cdr_{cdr}_pred'] = mod_cplx.get_cdr(cdr).get_seq()

    # 准备输出数据格式
    output_data = {
        'filename': mod_pdb,
        'metrics': results,
        'sequences': sequences
    }

    return output_data


def test_single_checkpoint(ckpt_path, test_set_path, save_dir, batch_size=32, num_workers=4, gpu=0):
    """测试单个checkpoint"""
    checkpoint_name = os.path.splitext(os.path.basename(ckpt_path))[0]
    print_log(f'\n{"="*60}')
    print_log(f'Testing checkpoint: {checkpoint_name}')
    print_log(f'{"="*60}')
    
    # 创建该checkpoint的保存目录
    ckpt_save_dir = create_save_dir(save_dir, checkpoint_name)
    print_log(f'Results will be saved to: {ckpt_save_dir}')

    # 加载模型 (简化版本，专门用于igformer)
    try:
        model = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except ValueError as e:
        if "Default process group has not been initialized" in str(e):
            # 设置环境变量并初始化分布式进程组
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['RANK'] = '0'
            os.environ['WORLD_SIZE'] = '1'
            
            import torch.distributed as dist
            if not dist.is_initialized():
                dist.init_process_group(backend='nccl', init_method='env://', world_size=1, rank=0)
            model = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            dist.destroy_process_group()
        else:
            raise e
    
    # 如果是DDP模型，提取module
    if hasattr(model, 'module'):
        model = model.module
    
    # 添加安全全局变量以支持PyTorch 2.6的weights_only模式
    try:
        torch.serialization.add_safe_globals([type(model)])
    except Exception:
        pass
    
    device = torch.device('cpu' if gpu == -1 else f'cuda:{gpu}')
    model.to(device)
    model.eval()

    print_log(f'Model type: {type(model)}')
    cdr_type = model.cdr_type
    print_log(f'CDR type: {cdr_type}')
    print_log(f'Paratope definition: {model.paratope}')

    # 加载测试数据集
    test_set = E2EDataset(test_set_path, cdr=cdr_type)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers, collate_fn=E2EDataset.collate_fn
    )

    summary_items = []
    idx = 0

    print_log("Starting antibody generation...")
    for batch in tqdm(test_loader, desc=f"Generating antibodies for {checkpoint_name}"):
        with torch.no_grad():
            for k in batch:
                if hasattr(batch[k], 'to'):
                    batch[k] = batch[k].to(device)

            del batch['xloss_mask']
            X, S, pmets = model.sample(**batch)
            X, S, pmets = X.tolist(), S.tolist(), pmets.tolist()

            X_list, S_list = [], []
            cur_bid = -1
            batch_id = batch.get('bid', None)
            if batch_id is None:
                lengths = batch['lengths']
                batch_id = torch.zeros_like(batch['S'])
                batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
                batch_id.cumsum_(dim=0)

            for i, bid in enumerate(batch_id):
                if bid != cur_bid:
                    cur_bid = bid
                    X_list.append([])
                    S_list.append([])
                X_list[-1].append(X[i])
                S_list[-1].append(S[i])

        for i, (x, s) in enumerate(zip(X_list, S_list)):
            ori_cplx = test_set.data[idx]
            cplx = to_cplx(ori_cplx, x, s)

            pdb_id = cplx.get_id().split('(')[0]
            mod_pdb = os.path.join(ckpt_save_dir, pdb_id + '.pdb')
            cplx.to_pdb(mod_pdb)

            ref_pdb = os.path.join(ckpt_save_dir, pdb_id + '_original.pdb')
            ori_cplx.to_pdb(ref_pdb)

            summary_items.append({
                'mod_pdb': mod_pdb,
                'ref_pdb': ref_pdb,
                'H': cplx.heavy_chain,
                'L': cplx.light_chain,
                'A': cplx.antigen.get_chain_names(),
                'cdr_type': cdr_type,
                'pdb': pdb_id,
                'pmetric': pmets[i]
            })
            idx += 1

    # 保存summary.json
    summary_file = os.path.join(ckpt_save_dir, 'summary.json')
    with open(summary_file, 'w') as fout:
        fout.writelines([json.dumps(item) + '\n' for item in summary_items])

    print_log(f'Summary of generated complexes written to {summary_file}')
    print_log(f'Generated {len(summary_items)} antibodies successfully')

    # 开始计算指标
    print_log("Starting metric calculation...")
    
    metric_inputs, pdbs = [], [item['pdb'] for item in summary_items]
    pmets = []
    for item in summary_items:
        keys = ['mod_pdb', 'ref_pdb', 'H', 'L', 'A', 'cdr_type']
        inputs = [item[key] for key in keys]
        if 'sidechain' in item:
            inputs.append(item['sidechain'])
        metric_inputs.append(inputs)
        pmets.append(item['pmetric'])

    if num_workers > 1:
        metrics = process_map(cal_metrics, metric_inputs, max_workers=num_workers)
    else:
        metrics = [cal_metrics(inputs) for inputs in tqdm(metric_inputs, desc="Calculating metrics")]

    # 保存指标结果
    metrics_file = os.path.join(ckpt_save_dir, 'metrics_output.txt')
    with open(metrics_file, 'w') as fout:
        # 写入每个模型的指标
        for metric in metrics:
            fout.write(f"Filename: {metric['filename']}\n")
            fout.write("Metrics:\n")
            for metric_name, value in metric['metrics'].items():
                fout.write(f"\t{metric_name}: {value}\n")
            fout.write("Sequences:\n")
            for seq_name, seq in metric['sequences'].items():
                fout.write(f"\t{seq_name}: {seq}\n")
            fout.write("\n")

        # 计算并写入总体指标
        fout.write("Total Metrics (Averaged across all models):\n")
        for name in metrics[0]['metrics']:
            vals = [item['metrics'][name] for item in metrics]
            avg_value = sum(vals) / len(vals)
            fout.write(f"\t{name}: {avg_value}\n")
        
        # 计算并写入总体的 Pearson 相关系数
        fout.write("\nPearson Correlation between development metric and evaluated metrics:\n")
        for name in metrics[0]['metrics']:
            vals = [item['metrics'][name] for item in metrics]
            corr = np.corrcoef(pmets, vals)[0][1]
            fout.write(f"\t{name}: Pearson Correlation = {corr}\n")
        
        fout.write("\n")

    print_log(f'Metrics calculation completed. Results saved to {metrics_file}')
    
    # 打印总体结果摘要
    print_log(f"\n=== RESULTS SUMMARY for {checkpoint_name} ===")
    summary_results = {}
    for name in metrics[0]['metrics']:
        vals = [item['metrics'][name] for item in metrics]
        avg_value = sum(vals) / len(vals)
        summary_results[name] = avg_value
        print_log(f"{name}: {avg_value:.4f}")
    
    print_log(f"Results saved to: {ckpt_save_dir}")
    
    return summary_results, checkpoint_name


def test_all_checkpoints(checkpoint_dir, test_set_path, save_dir, batch_size=32, num_workers=4, gpu=0):
    """测试checkpoint目录中的所有ckpt文件"""
    # 查找所有.ckpt文件
    ckpt_pattern = os.path.join(checkpoint_dir, "*.ckpt")
    ckpt_files = glob.glob(ckpt_pattern)
    
    if not ckpt_files:
        print_log(f"No checkpoint files found in {checkpoint_dir}", level='ERROR')
        return
    
    # 按文件名排序
    ckpt_files.sort()
    
    print_log(f"Found {len(ckpt_files)} checkpoint files:")
    for i, ckpt in enumerate(ckpt_files, 1):
        print_log(f"  {i}. {os.path.basename(ckpt)}")
    
    # 创建总体结果保存目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    overall_save_dir = os.path.join(save_dir, f"all_checkpoints_test_{timestamp}")
    os.makedirs(overall_save_dir, exist_ok=True)
    
    # 存储所有checkpoint的结果
    all_results = []
    
    # 逐个测试每个checkpoint
    for i, ckpt_path in enumerate(ckpt_files, 1):
        try:
            print_log(f"\n{'='*80}")
            print_log(f"Testing checkpoint {i}/{len(ckpt_files)}: {os.path.basename(ckpt_path)}")
            print_log(f"{'='*80}")
            
            summary_results, checkpoint_name = test_single_checkpoint(
                ckpt_path, test_set_path, overall_save_dir, batch_size, num_workers, gpu
            )
            
            all_results.append({
                'checkpoint': checkpoint_name,
                'checkpoint_path': ckpt_path,
                'metrics': summary_results
            })
            
        except Exception as e:
            print_log(f"Error testing checkpoint {ckpt_path}: {e}", level='ERROR')
            all_results.append({
                'checkpoint': os.path.splitext(os.path.basename(ckpt_path))[0],
                'checkpoint_path': ckpt_path,
                'error': str(e)
            })
    
    # 保存所有结果的汇总
    summary_file = os.path.join(overall_save_dir, 'all_checkpoints_summary.json')
    with open(summary_file, 'w') as fout:
        json.dump(all_results, fout, indent=2)
    
    # 创建CSV格式的结果汇总
    csv_file = os.path.join(overall_save_dir, 'all_checkpoints_results.csv')
    if all_results and 'metrics' in all_results[0]:
        # 获取所有指标名称
        metric_names = list(all_results[0]['metrics'].keys())
        
        with open(csv_file, 'w', newline='') as fout:
            writer = csv.writer(fout)
            # 写入表头
            header = ['Checkpoint'] + metric_names
            writer.writerow(header)
            
            # 写入数据
            for result in all_results:
                if 'metrics' in result:
                    row = [result['checkpoint']] + [result['metrics'].get(name, 'N/A') for name in metric_names]
                    writer.writerow(row)
                else:
                    row = [result['checkpoint']] + ['ERROR'] * len(metric_names)
                    writer.writerow(row)
    
    # 打印最终汇总
    print_log(f"\n{'='*80}")
    print_log("FINAL SUMMARY - ALL CHECKPOINTS")
    print_log(f"{'='*80}")
    
    if all_results and 'metrics' in all_results[0]:
        # 计算每个指标的平均值
        metric_names = list(all_results[0]['metrics'].keys())
        print_log(f"{'Checkpoint':<30} {'AAR':<8} {'CAAR':<8} {'RMSD(CA)':<10} {'RMSD(CA) CDRH3':<15} {'TMscore':<8} {'LDDT':<8} {'DockQ':<8}")
        print_log("-" * 95)
        
        for result in all_results:
            if 'metrics' in result:
                metrics = result['metrics']
                print_log(f"{result['checkpoint']:<30} {metrics.get('AAR', 0):<8.4f} {metrics.get('CAAR', 0):<8.4f} {metrics.get('RMSD(CA)', 0):<10.4f} {metrics.get('RMSD(CA) CDRH3', 0):<15.4f} {metrics.get('TMscore', 0):<8.4f} {metrics.get('LDDT', 0):<8.4f} {metrics.get('DockQ', 0):<8.4f}")
            else:
                print_log(f"{result['checkpoint']:<30} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<15} {'ERROR':<8} {'ERROR':<8} {'ERROR':<8}")
    
    print_log(f"\nAll results saved to: {overall_save_dir}")
    print_log(f"Summary JSON: {summary_file}")
    print_log(f"Results CSV: {csv_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Test all checkpoints in a directory for Igformer')
    parser.add_argument('--checkpoint_dir', type=str, required=True, 
                        help='Directory containing checkpoint files (.ckpt)')
    parser.add_argument('--test_set', type=str, required=True, 
                        help='Path to test set')
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save test results (relative to current working directory)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers to use')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    return parser.parse_args()


if __name__ == '__main__':
    setup_seed(4)
    args = parse_args()
    test_all_checkpoints(
        args.checkpoint_dir, 
        args.test_set, 
        args.save_dir, 
        args.batch_size, 
        args.num_workers, 
        args.gpu
    )


# 使用示例：
# python test_all_checkpoints.py --checkpoint_dir my_checkpoints/models_single_cdr_design/version_10/checkpoint  --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 16 --gpu 6


#  python test_all_checkpoints.py --checkpoint_dir my_checkpoints/models_single_cdr_design/version_0/checkpoint --test_set all_data/RAbD/test_fixed.json --save_dir results --batch_size 2 --gpu 6
