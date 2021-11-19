import numpy as np 

def save_edgelist(edgelist, fpath, sep='\t'):
    #for df: turn it into an edgelist first:
    # edgelist = edge_df.reset_index().values.tolist()
    with open(fpath, 'w') as f:
        for item in edgelist:
            f.write("%s%s%s%s%s\n" %(item[0],sep,item[1],sep,item[2]))
    print('Finished saving edgelist!')

def get_klpq_numpy(p_probs, q_probs):
    p = np.array(p_probs)
    q = np.array(q_probs)
    div = np.multiply(p, np.log(np.divide(p,q)))
    kl_div = np.sum(div)

    return kl_div

def get_klpq_div(p_probs, q_probs):
    kl_div = 0.0
    
    for pi, qi in zip(p_probs, q_probs):
        kl_div += pi*np.log(pi/qi)
    
    return kl_div

def normalize(v):
    norm = np.linalg.norm(v,ord=1)
    if norm == 0: 
        return v
    return v / norm