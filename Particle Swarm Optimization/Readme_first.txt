
"Do_PSO" is the function call

Inputs Parameters:
"""
    
    ----------
    N : Input particles and it'svector
    T : Number of iteration
    w : inertia weight
    c1 : cognitive term constant
    c2 : social term constant
    F : Fitness function(where we can define our own)
    lb : Lower Bound of Decision Vector 
    ub : Upper Bound of Decision Vector
    
    Governing Equation:
        Xi = Xi + Vi
        Vi = wVi + c1*r1*(P_best - Xi) + c2*r2*(g_best - Xi)

    Returns
    -------
    global_vector,global_best,N_updated, F_updated,N
    """

