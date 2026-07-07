import argparse
import torch
import pandas as pd
from copy import deepcopy
import os

from eval.vsun.relax import relax_structures
from eval.vsun.utils import maybe_get_missing_columns, COLUMNS_COMPUTATIONS, filter_prerelaxed
from eval.vsun.structure_summary import get_metrics_structure_summaries
from eval.vsun.reference.reference_dataset import ReferenceDataset
from eval.vsun.reference.presets import mp_20, alex_mp_20
from eval.vsun.evaluator import Evaluator
from utils.set_logger import get_logger
from utils.config import DictAction
from utils.crys_utils import dict_to_valid_struct

from pymatgen.core import Structure
from pymatgen.entries.compatibility import MaterialsProject2020Compatibility


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mp_20')
    parser.add_argument('--conf_new', nargs='+', action=DictAction)
    args = parser.parse_args()
    args.task = "base"
    args.log = os.path.join('logs', args.task, args.dataset)
    args.mlff = 'mattersim'
    args.max_relax_steps = 500
    args.stable_delta = 0.1

    if args.dataset == "mp_20":
        num_turns = 400
        args.ehull_ref = "mp_20"
        reference = mp_20()
    elif args.dataset == "alex_mp_20":
        num_turns = 100
        args.ehull_ref = "alex_mp_20"
        reference = alex_mp_20()

    args.eval_path = os.path.join(args.log, 'eval')
    saved_path = os.path.join(args.log, "gen")
    os.makedirs(args.eval_path, exist_ok=True)

    logger = get_logger(path=os.path.join(args.eval_path, 'eval_vsun.log'))
    args.logger = logger
    args.logger.info(args)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
    torch.backends.cudnn.benchmark = True

    for turn in range(num_turns):
        try:
            args.logger.info("Round "+str(turn))
            gen = torch.load(os.path.join(saved_path, "gen_"+str(turn)+".pt"), weights_only=False)
            args.num_samples = len(gen)

            ########## relax structure ##########
            args.logger.info("Relaxing the generated structures, using "+args.mlff)
            relax_path = os.path.join(args.eval_path, str(turn), "relaxed_"+args.mlff+".json")
            os.makedirs(os.path.join(args.eval_path, str(turn)), exist_ok=True)
            if not os.path.exists(relax_path):
                ########## convert to structures ##########
                args.logger.info("Converting generated tensors into Structures...")
                structures, _ = dict_to_valid_struct(gen)
                df = relax_structures(structures, device=device, steps=args.max_relax_steps, mlff=args.mlff)
                df.to_json(relax_path)
        
            ########## change to Reference dataset (MatterGen) ##########
            df = pd.read_json(relax_path)
            df = maybe_get_missing_columns(df, COLUMNS_COMPUTATIONS)
            df = filter_prerelaxed(df)

            relaxed_structures, relaxed_energies = [Structure.from_dict(s) for s in df["structure"]], df["e_relax"].tolist()
            structure_summaries = get_metrics_structure_summaries(
                    structures=relaxed_structures,
                    energies=relaxed_energies,
                    energy_correction_scheme=MaterialsProject2020Compatibility(),
                )
            data_entries = [deepcopy(s.entry) for s in structure_summaries]
            for i, e in enumerate(data_entries):
                e.entry_id = i
            gen_ref = ReferenceDataset.from_entries("data_entries", data_entries)

            ########## compute VSUN ##########
            args.logger.info("Computing Metrics ...")
            evaluator = Evaluator(args, gen_ref, reference)
            u_indicator = evaluator.get_uniqueness()
            n_indicator = evaluator.get_novelty()
            un_indicators = torch.BoolTensor(u_indicator * n_indicator)
            indicators = {
                "vu": u_indicator,
                "vn": n_indicator,
                "vun": un_indicators
            }
            torch.save(indicators, os.path.join(args.eval_path, str(turn), "indicators_"+args.mlff+"_"+args.ehull_ref+".pt"))
        except:
            pass

