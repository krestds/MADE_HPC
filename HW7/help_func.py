from SALib.sample import saltelli

def sample_func_(inp_arg):
    
    (problem, n_samples, i) = inp_arg
    print(f'process #{i}')
    
    return saltelli.sample(problem, n_samples, i*n_samples)
