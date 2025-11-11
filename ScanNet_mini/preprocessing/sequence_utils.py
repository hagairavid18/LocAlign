import numpy as np
import pandas as pd
from numba import prange, njit
from scipy.interpolate import interp1d
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utilities.paths import path2hhblits, path2sequence_database,path2mmseqs, path2mmseqsdatabases, path2mmseqstmp,mmseqsdatabase
from time import sleep
import subprocess,fileinput,shutil

curr_float = np.float32
curr_int = np.int16

aa = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
      'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '-']
aadict = {aa[k]: k for k in range(len(aa))}

aadict['X'] = len(aa)
aadict['B'] = len(aa)
aadict['Z'] = len(aa)
aadict['O'] = len(aa)
aadict['U'] = len(aa)

for k, key in enumerate(
        ['a', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'y']):
    aadict[key] = aadict[aa[k]]
aadict['x'] = len(aa)
aadict['b'] = len(aa)
aadict['z'] = -1
aadict['.'] = -1


# def seq2num(string):
#     if type(string) == str:
#         return np.array([aadict[x] for x in string])[np.newaxis, :]
#     elif type(string) == list:
#         return np.array([[aadict[x] for x in string_] for string_ in string])

def seq2num(string):
    if type(string) in [str, np.str_]:
        return np.array([aadict[x] for x in string])[np.newaxis, :]
    elif type(string) in [list, np.ndarray]:
        return np.array([[aadict[x] for x in string_] for string_ in string])


def num2seq(num):
    if num.ndim == 1:
        return ''.join([aa[min(x, len(aa) - 1)] for x in num])
    else:
        return [''.join([aa[min(x, len(aa) - 1)] for x in num_seq]) for num_seq in num]


def load_FASTA(filename, with_labels=True, numerical=True, remove_insertions=True, drop_duplicates=True):
    count = 0
    current_seq = ''
    all_seqs = []
    if with_labels:
        all_labels = []
    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '>':
                all_seqs.append(current_seq)
                current_seq = ''
                if with_labels:
                    all_labels.append(line[1:].replace(
                        '\n', '').replace('\r', ''))
            else:
                current_seq += line.replace('\n', '').replace('\r', '')
                count += 1
            if remove_insertions:
                current_seq = ''.join(
                    [x for x in current_seq if not (x.islower() | (x == '.'))])

        all_seqs.append(current_seq)
        if numerical:
            all_seqs = np.array(list(
                map(lambda x: [aadict[y] for y in x], all_seqs[1:])), dtype=curr_int, order="c")
        else:
            all_seqs = np.array(all_seqs[1:])
    if drop_duplicates:
        all_seqs = pd.DataFrame(all_seqs).drop_duplicates()
        if with_labels:
            all_labels = np.array(all_labels)[all_seqs.index]
        all_seqs = np.array(all_seqs)

    if with_labels:
        return all_seqs, np.array(all_labels)
    else:
        return all_seqs


@njit(parallel=False, cache=True, nogil=False)
def weighted_average(config, weights, q):
    B = config.shape[0]
    N = config.shape[1]
    out = np.zeros((N, q), dtype=curr_float)
    for b in prange(B):
        for n in prange(N):
            out[n, config[b, n]] += weights[b]
    out /= weights.sum()
    return out


@njit(parallel=True, cache=True)
def count_neighbours(MSA, threshold=0.1, remove_gaps=False):  # Compute reweighting
    B = MSA.shape[0]
    num_neighbours = np.ones(B, dtype=curr_int)
    for b1 in prange(B):
        for b2 in prange(B):
            if b2 > b1:
                if remove_gaps:
                    are_neighbours = ((MSA[b1] != 20) * (MSA[b2] != 20) * (MSA[b1] != MSA[b2])).sum() / (
                                (MSA[b1] != 20) * (MSA[b2] != 20)).sum() < threshold
                else:
                    are_neighbours = (MSA[b1] != MSA[b2]).mean() < threshold
                num_neighbours[b1] += are_neighbours
                num_neighbours[b2] += are_neighbours
    return np.asarray(num_neighbours, dtype=curr_int)


def get_focusing_weights(all_sequences, all_weights, WT, targets_Beff, step=0.5):
    homology = 1 - (all_sequences == all_sequences[WT]).mean(-1)
    Beff = all_weights.sum()
    targets_Beff = np.array(targets_Beff)
    Beff_min = targets_Beff.min()
    all_focusing_weights = np.ones(
        [len(all_weights), len(targets_Beff)], dtype=np.float32)

    if Beff_min < Beff:  # Attempt focusing.
        # First, determine the largest focusing coefficient to be applied.
        Beff_current = Beff
        focusing = 0.0
        all_focusings = [0.0]
        all_Beff = [Beff]
        while Beff_current > Beff_min:
            focusing += step
            focusing_weights = np.exp(-focusing * homology)
            Beff_current = (all_weights * focusing_weights).sum()
            all_focusings.append(focusing)
            all_Beff.append(Beff_current)
        # Next interpolate to determine the correct focusing coefficients to be applied to each target.
        f = interp1d(all_Beff, all_focusings, bounds_error=False)
        target_focusings = f(targets_Beff)
        target_focusings[targets_Beff > Beff] = 0.
        for l, target_focusing in enumerate(target_focusings):
            all_focusing_weights[:, l] = np.exp(-target_focusing * homology)
    return all_focusing_weights


def conservation_score(PWM, Beff, Bvirtual=5):
    eps = Bvirtual / (Bvirtual + Beff * (1 - PWM[:, -1]))
    PWM = PWM[:, :-1].copy()
    PWM /= PWM.sum(-1)[:, np.newaxis]
    PWM = eps[:, np.newaxis] / 20 + (1 - eps[:, np.newaxis]) * PWM
    conservation = np.log(20) - (- np.log(PWM) * PWM).sum(-1)
    return conservation


def compute_PWM(location, gap_threshold=0.4,
                neighbours_threshold=0.1, Beff=500, WT=0, nmax=10000, scaled=False):
    if not isinstance(Beff, list):
        Beff = [Beff]

    nBeff = len(Beff)

    all_sequences, all_labels = load_FASTA(
        location, remove_insertions=True, with_labels=True, drop_duplicates=True)
    sequences_with_few_gaps = (all_sequences == 20).mean(-1) < gap_threshold
    all_sequences = all_sequences[sequences_with_few_gaps]
    all_labels = all_labels[sequences_with_few_gaps]

    if len(all_sequences) >= nmax:
        d2wt = (all_sequences != all_sequences[WT:WT + 1]).mean(-1)
        subset = np.argsort(d2wt)[:nmax]
        all_sequences = all_sequences[subset]
        all_labels = all_labels[subset]

    all_weights = 1.0 / np.maximum(1.,count_neighbours(all_sequences, threshold=neighbours_threshold).astype(float) )

    ambiguous_residues = np.nonzero(all_sequences == 21)
    if len(ambiguous_residues[0]) > 0:
        all_sequences[ambiguous_residues[0], ambiguous_residues[1]] = 20
        PWM = weighted_average(all_sequences, all_weights.astype(curr_float), 21)
        consensus = np.argmax(PWM, axis=-1)
        all_sequences[ambiguous_residues[0], ambiguous_residues[1]] = consensus[ambiguous_residues[1]]

    all_focusing_weights = get_focusing_weights(
        all_sequences, all_weights, WT, Beff)
    all_weights_focused = all_weights[:, np.newaxis] * all_focusing_weights
    all_weights_focused /= all_weights_focused.mean(0)

    PWM = np.zeros([all_sequences.shape[-1], 21, nBeff], dtype=curr_float)
    for n in range(nBeff):
        PWM[:, :, n] = weighted_average(
            all_sequences, all_weights_focused[:, n].astype(curr_float), 21)
    if scaled:
        Beff = all_weights.sum()
        for n in range(nBeff):
            conservation = conservation_score(PWM[:, :, n], Beff, Bvirtual=5)
            PWM[:, :, n] *= conservation[:, np.newaxis]
    if nBeff == 1:
        PWM = PWM[:, :, 0]
    return PWM


def call_hhblits(sequence, output_alignment, 
                        filtermsa = True,
                        cov = 0.6,
                        qid = 0.35,
                        maxseqid = 0.95,
                        path2hhblits=path2hhblits, path2sequence_database=path2sequence_database,
                        overwrite=False, cores=6,iterations=2,MSA=None):

    query_file = output_alignment[:-6] + '_query.fasta'
    output_file = output_alignment[:-6] + 'metadata.txt'

    if not overwrite:
        if os.path.exists(output_alignment) & os.path.exists(output_file):
            print('File %s already exists. Not recomputing' %
                  output_alignment)
            return output_alignment
    if MSA is not None:
        os.system('scp %s %s' % (MSA, query_file))
    else:
        with open(query_file, 'w') as f:
            f.write('>>WT\n')
            f.write(sequence)
            
    cmd = [path2hhblits, '-cpu', str(cores), '-n', str(iterations), '-i', query_file.replace(' ', '\ '), '-o', output_file.replace(' ', '\ '), '-oa3m', output_alignment.replace(' ', '\ '), '-d', path2sequence_database]        
    
    
    
    if filtermsa:
        if cov<1:
            cov = cov * 100
        if maxseqid<1:
            maxseqid = maxseqid*100
        if qid<1:
            qid = qid*100
        cmd += ['-id', str(maxseqid), '-cov', str(cov), '-qid', str(qid)]
    else:
        cmd += ['-all']
        
    # print(cmd)    
    
    # cmd = '%s -cpu %s -all -n %s -i %s -o %s -oa3m %s -d %s' % (
    #     path2hhblits, cores, iterations, query_file.replace(' ', '\ '), output_file.replace(' ', '\ '),
    #     output_alignment.replace(' ', '\ '), path2sequence_database)
    t = time.time()
    subprocess.call(cmd)
    os.system('rm %s' % query_file.replace(' ', '\ '))
    os.system('rm %s' % output_file.replace(' ', '\ '))
    print('Called hhblits finished: Duration %.2f s' % (time.time() - t))
    return output_alignment


def call_mmseqs(
        sequence,
        output_file,        
        MSA = None,
        database = mmseqsdatabase,
        cores = 6,
        filtermsa = True,
        cov = 0.6,
        qid = 0.35,
        maxseqid = 0.95,
        gapopen = 11,
        gapextend = 1,
        s = 5.7000,
        num_iterations = 1,
        maxseqs = 10000,
        overwrite = False,
        report=None,
        path2mmseqs=path2mmseqs,
        path2mmseqsdatabases=path2mmseqsdatabases,
        path2mmseqstmp=path2mmseqstmp,
        nattempts=0):
    
    try:    
        if MSA is not None:
            input_file = MSA
            delete_input = False
        else:
            input_file = output_file.split('.')[0] + '_input.fasta'
            delete_input = True
            with open(input_file,'w') as f:
                f.write('>query\n')
                f.write(f'{sequence}\n')    
        
        t = time.time()
        if cov>1:
            cov = cov/100.
        if not overwrite:
            if os.path.exists(output_file):
                print('File %s already exists. Not recomputing' %output_file,file=report)
                return output_file

        ninputs = sum([line.startswith('>') for line in open(input_file,'r').readlines()])

        '''
        Source: https://github.com/soedinglab/MMseqs2/issues/693
        '''
        tmp_folder = '.'.join(output_file.split('.')[:-1]) + '/'
        os.makedirs(tmp_folder,exist_ok=True)
        tmp_input_file = os.path.join(tmp_folder,'input')
        tmp_output_file = os.path.join(tmp_folder, 'output')
        tmp_output_file2 = os.path.join(tmp_folder, 'output2')

        commands = [
            [path2mmseqs, 'createdb', input_file, tmp_input_file],
            [path2mmseqs, 'search', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file, path2mmseqstmp,
            '-s',str(s),'--cov', str(cov),'--cov-mode','2','--diff',str(maxseqs), '--qid', str(qid), '--max-seq-id', str(maxseqid),'--gap-open', str(gapopen), '--gap-extend', str(gapextend), '--threads',str(cores),'--num-iterations',str(num_iterations),
            '--max-seqs',str(maxseqs),'-e',str(0.1),'-a',"--db-load-mode", '2','--filter-min-enable','5'],
            [path2mmseqs, 'result2msa', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file,tmp_output_file2,
            '--filter-msa',str(int(filtermsa)), '--cov',str(cov),'--diff',str(maxseqs),'--qid', str(qid), '--max-seq-id',str(maxseqid),'--msa-format-mode','5','--gap-open', str(gapopen), '--gap-extend',str(gapextend), '--threads',str(cores),"--db-load-mode", '2'],
            [path2mmseqs, 'convertalis', tmp_input_file, os.path.join(path2mmseqsdatabases,database), tmp_output_file,tmp_output_file +'.tab','--format-output',"target,theader", '--threads',str(cores),"--db-load-mode", '2'],
            [path2mmseqs, 'unpackdb', tmp_output_file2, tmp_folder, '--unpack-name-mode', '0', '--threads',str(cores)]
        ]
        for command in commands:
            # print(' '.join(command))
            subprocess.call(command)
        
        try:
            table_labels = pd.read_csv(tmp_output_file +'.tab',sep='\t',header=None,index_col=0).drop_duplicates()
        except:
            table_labels = None

        for n in range(ninputs):
            if ninputs == 1:
                output_file_ = output_file
            else:
                output_file_ = output_file.split('.fasta')[0] + '_%s.fasta'%n
            os.rename(os.path.join(tmp_folder,str(n) ),output_file_)
            with fileinput.input(files=output_file_, inplace=True) as f:
                for line in f:
                    if line.startswith('>'):
                        try:
                            newlabel = table_labels.loc[line[1:-1]].item()
                        except:
                            newlabel = 'na|%s|' % line[1:-1]
                        newline = '>' + newlabel + '\n'
                    else:
                        newline = line
                    print(newline, end='')
            assert os.path.exists(output_file_)
        subprocess.call(['rm', '-r', tmp_folder])
        if delete_input and os.path.exists(input_file):
            os.remove(input_file)
        print('Called mmseqs finished: Duration %.2f s' % (time.time() - t),file=report)
        return output_file
    except Exception as e:
        print(f'Call mmseqs failed for {output_file} error {e} after {nattempts} attempts')
        if os.path.exists(tmp_folder):
            shutil.rmtree(tmp_folder)
        if delete_input and os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)
        if nattempts < 5:
            return call_mmseqs(
                sequence,
                output_file,        
                MSA = MSA,
                database = database,
                cores = cores,
                filtermsa = filtermsa,
                cov = cov,
                qid = qid,
                maxseqid = maxseqid,
                gapopen = gapopen,
                gapextend = gapextend,
                s = s,
                num_iterations = num_iterations,
                maxseqs = maxseqs,
                overwrite = overwrite,
                report=report,
                path2mmseqs=path2mmseqs,
                path2mmseqsdatabases=path2mmseqsdatabases,
                path2mmseqstmp=path2mmseqstmp,nattempts=nattempts+1)
        else:
            raise ValueError(f'Call mmseqs failed for {output_file} error {e} after {nattempts} attempts')
            
            
            
        
        
        

#
# def call_mmseq_server(sequence,output_alignment):
#     # From https://github.com/soedinglab/MMseqs2-App/blob/master/docs/api_example.py
#
#     # submit a new job
#     ticket = post('https://search.mmseqs.com/api/ticket', {
#                 'q'
#                 : '>FASTA\n%s\n'%sequence,
#                 'database[]' :["uniclust30_2017_10_seed"],
#                 'mode':'all',
#             }).json()
#
#     # poll until the job was successful or failed
#     repeat = True
#     while repeat:
#         status = get('https://search.mmseqs.com/api/ticket/' + ticket['id']).json()
#         if status['status'] == "ERROR":
#             # handle error
#             sys.exit(0)
#
#         # wait a short time between poll requests
#         sleep(1)
#         repeat = status['status'] != "COMPLETE"
#
#     # get all hits for the first query (0)
#     result = get('https://search.mmseqs.com/api/result/' + ticket['id'] + '/0').json()
#     # print pairwise alignment of first hit of first database
#     print(result['results'][0]['alignments'][0]['qAln'])
#     print(result['results'][0]['alignments'][0]['dbAln'])
#
#     # download blast compatible result archive
#     download = get('https://search.mmseqs.com/api/result/download/' + ticket['id'], stream=True)
#     with open('result.tar.gz', 'wb') as fd:
#         for chunk in download.iter_content(chunk_size=128):
#             fd.write(chunk)
#     return
