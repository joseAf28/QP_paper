import sympy as sp
import numpy as np
import cmath


GA = sp.Symbol('GA', commutative=False); GR = sp.Symbol('GR', commutative=False); GK = sp.Symbol('GK', commutative=False);
gA = sp.Symbol('gA', commutative=False); gR = sp.Symbol('gR', commutative=False); gK = sp.Symbol('gK', commutative=False);
dot = sp.Symbol('.', commutative=False);
F = sp.Symbol('F', commutative=False)

Gamma = sp.Symbol('Gamma', commutative=True)


def get_green_const(term):
    
    const = []
    green = []
    
    for s1 in term.as_ordered_terms():
        factors = s1.as_ordered_factors()
        
        green_aux = []
        const_aux = []
        
        for f in factors:
            if f.has(GR) or f.has(GA) or f.has(GK) or f.has(gK) or f.has(dot):
                green_aux.append(f)
            else:
                const_aux.append(f)
        
        if const_aux == []:
            const_aux = [sp.Integer(1)]
        
        const.append(const_aux)
        green.append(green_aux)
        
    return const, green


def get_functions_green(term_green):
    
    vec_str = [str(v) if v != dot else '_' for v in term_green]
    vec_str_ordered = []
    
    for i, item in enumerate(vec_str):
        if item == '_':
            str1 = vec_str[:i]
            str2 = vec_str[i+1:]
            
            sorted_function = ''.join(sorted(str1)) + '_' + ''.join(sorted(str2))
            sorted_function_equiv = ''.join(sorted(str2)) + '_' + ''.join(sorted(str1))
            
            vec_str_ordered.append((sorted_function, '+1'))
            # vec_str_ordered.append((sorted_function_equiv, '-1'))
        
        else:
            pass
        
    return vec_str_ordered



def get_functions_green_FCS(term_green):
    
    vec_str = [str(v) if v != dot else '_' for v in term_green]
    vec_str_ordered = []
    
    for i, item in enumerate(vec_str):
        if item == '_':
            str1 = vec_str[:i]
            str2 = vec_str[i+1:]
            
            sorted_function = ''.join(sorted(str1)) + ''.join(sorted(str2))
            sorted_function_equiv = ''.join(sorted(str2)) + ''.join(sorted(str1))
            
            vec_str_ordered.append((sorted_function, '+1'))
            # vec_str_ordered.append((sorted_function_equiv, '-1'))
        
        else:
            pass
        
    return vec_str_ordered
