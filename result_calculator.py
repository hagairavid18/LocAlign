homology_counts = {
    '0': 183,
    '1': 542,
    '2': 436,
    '3': 261,
    '4': 831
}   

ligand_counts = {
    '0': 460,
    '1': 1131,
    '2': 722,
    '3': 483,
    '4': 537
}

ligand_exp_results = {
    "split7-10gnn-recycling2-lambda05": {'0': 0.0739, '1': 0.0274, '2': 0.1288, '3': 0.0745, '4': 0.730},
    "baseline-2409-TMalign": {'0': 0.0369, '1': 0.0097, '2': 0.0152, '3': 0.0207, '4': 0.7132},
    "baseline-2409-dali": {'0': 0.0391, '1': 0.0106, '2': 0.0110, '3': 0.0289, '4': 0.6909},
    "baseline-2409-Softalign-sw": {'0': 0.0108, '1': 0.0035, '2': 0.0000, '3': 0.0124, '4': 0.5643},
    "split7-0gnn-recycling2-lambda05": {'0': 0.0543, '1': 0.0291, '2': 0.0886, '3': 0.0559, '4': 0.7002},
    "split7-10gnn-recycling10-lambda05-no-esm" : {'0': 0.0478, '1': 0.0106, '2': 0.0346, '3': 0.0310, '4': 0.694},
    "split7-10gnn-recycling0-lambda05": {'0': 0.0739, '1': 0.0327, '2': 0.1053, '3': 0.0703, '4': 0.7225},
    "split7-10gnn-recycling2-lambda0": {'0': 0.0717, '1': 0.0309, '2': 0.1246, '3': 0.0662, '4': 0.702},
}

homology_exp_results = {
    "split7-10gnn-recycling2-lambda05": {'0': 0.1913, '1': 0.2011, '2': 0.2798, '3': 0.0919, '4': 0.7593},
    "baseline-2409-TMalign": {'0': 0.1038, '1': 0.0350, '2': 0.0160, '3': 0.0268, '4': 0.5993},
    "baseline-2409-dali": {'0': 0.0983, '1': 0.0424, '2': 0.0183, '3': 0.0498, '4': 0.6077},
    "baseline-2409-Softalign-sw": {'0': 0.0163, '1': 0.0055, '2': 0.0048, '3': 0.0076, '4': 0.3430},
    "split7-0gnn-recycling2-lambda05": {'0': 0.1913, '1': 0.1771, '2': 0.2385, '3': 0.0459, '4': 0.7509},
    "split7-10gnn-recycling10-lambda05-no-esm" : {'0': 0.1694, '1': 0.0664, '2': 0.0389, '3': 0.0383, '4': 0.6835},
    "split7-10gnn-recycling0-lambda05": {'0': 0.1913, '1': 0.1273, '2': 0.1743, '3': 0.0728, '4': 0.7569},
    "split7-10gnn-recycling2-lambda0": {'0': 0.1967, '1': 0.2103, '2': 0.2729, '3': 0.0728, '4': 0.7557},
}



def calculate_results_up_to3(results, count_dict):
    weighted_sum = (results['0'] * count_dict['0'] + results['1'] * count_dict['1'] + results['2'] * count_dict['2'] + results['3'] * count_dict['3'])
    total = count_dict['0'] + count_dict['1'] + count_dict['2'] + count_dict['3']
    overall = weighted_sum / total
    return overall

def calculate_results_only4(results, count_dict):
    N_deg4 = count_dict['4']
    weighted_sum = (results['4'] * N_deg4)
    total = N_deg4
    overall = weighted_sum / total
    return overall

if __name__ == "__main__":
    
    print("Results on homology dataset:")
    for exp_name in homology_exp_results.keys():
        results = homology_exp_results[exp_name]
        overall_up_to3 = calculate_results_up_to3(results, homology_counts)
        overall_4 = calculate_results_only4(results, homology_counts)
        print(f"{exp_name}: up to 3: {overall_up_to3}, only 4: {overall_4}")

    print("\n\nResults on ligand dataset:")
    for exp_name in ligand_exp_results.keys():
        results = ligand_exp_results[exp_name]
        overall_up_to3 = calculate_results_up_to3(results, ligand_counts)
        overall_4 = calculate_results_only4(results, ligand_counts)
        print(f"{exp_name}: up to 3: {overall_up_to3}, only 4: {overall_4}")