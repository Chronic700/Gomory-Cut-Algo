# Initialize variables to store matrices A, b, and c
import numpy as np

error=1e-7

def inp():
    A = []
    b = []
    constraints = []
    c = []
    obj = 0

    # Read input from the file
    with open('input_ilp.txt', 'r') as file:
        # Read the entire contents of the file
        input_data = file.read()

    # Split input data into lines
    lines = input_data.strip().split('\n')

    if lines[1]=='maximize':
        obj=1
    else:
        obj=-1
        
    i=4
    while(len(lines[i])>0):
        A.append(list(map(float, lines[i].split(','))))
        i+=1
    
    actual_no_vars=len(A[0])

    i+=2

    while(len(lines[i])>0):
        b.append(float(lines[i]))
        i+=1

    i+=2

    while(len(lines[i])>0):
        if(lines[i][0]=='>'):
            constraints.append(">=")
        elif(lines[i][0]=='<'):
            constraints.append("<=")
        else:
            constraints.append("=")
        i+=1
        
    i+=2

    # print(lines[i])

    c=list(map(float, lines[i].split(',')))
    m = len(b)
    n = len(A[0])
    ineq=[]
    for i in range(0,m):
        if(constraints[i]!='='):
            ineq.append(i)
        if b[i]<0:
            b[i]=-1*b[i]
            for j in range(0,n):
                A[i][j] = -1* A[i][j]
            
            if(constraints[i]=='>='):
                constraints[i]='<='
                # ineq.append(i)
            elif(constraints[i]=='<='):
                constraints[i]='>='
                # ineq.append(i)
                

    j=0          
    for i in range(0,m):
        if(j==len(ineq)):
            for k in range(0,len(ineq)):
                A[i].append(0.0)
        elif(i!=ineq[j]):
            for k in range(0,len(ineq)):
                A[i].append(0.0)
        else:
            for k in range(0,j):
                A[i].append(0.0)
                
            if(constraints[i]=='>='):
                A[i].append(-1.0)
            else:
                A[i].append(1.0)
                
            for k in range(j+1,len(ineq)):
                A[i].append(0.0)
            j+=1
        
    for i in range(len(ineq)):
        c.append(0.0)
        
    if obj==1:              # max condition
        for i in range(len(c)):
            c[i] = -1*c[i]
        
                
    # # print the matrices and vectors
    # print("Matrix A:")
    # for row in A:
        # print(row)

    # print("\nVector b:")
    # print(b)

    # print("\nConstraints:")
    # print(constraints)

    # print("\nVector c:")
    # print(c)
    
    A=np.array(A) ; b=np.array(b); c=np.array(c)
    return A,b,c, obj, actual_no_vars

def simplex_algorithm(tableau,basis):
    m, n = tableau.shape
    # print(tableau)


    while np.any(tableau[0,1:] < -error):    ####      incorporate error       ####
        # pivot_col = np.argmin(tableau[0,1:])
        i=1
        while(tableau[0][i]>=-error):
            i+=1
        pivot_col = i

        if i>=n:
            break

        min_ratio_index = -1
        ratio=float('inf')
        for j in range(1,m):
            if(tableau[j][pivot_col]>error):
                cell_ratio=tableau[j][0]/tableau[j][pivot_col]
                if ratio==float('inf') or ratio>cell_ratio:
                    ratio = cell_ratio ; min_ratio_index = j
                    
        if min_ratio_index==-1:

            # print("Unbounded, ", "Col Number: ", i)
            
            return tableau, "unbounded",basis

        pivot_row = min_ratio_index
        
        # print("Previous basis: ", basis)
        # print(basis[pivot_row-1], " leaves ", pivot_col, " enters")

        basis[pivot_row-1]=pivot_col

        # print("New Basis: ", basis)
        

        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(m):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
        # print(tableau)

    return tableau, "optimal",basis

def phase_one(A, b):
    m, n = len(A),len(A[0])

    # Add artificial variables
    A = np.hstack((A, np.eye(m)))
    
    # Initialize objective coefficients
    c = np.zeros(n + m)
    c[n:] = 1  # Minimize the sum of artificial variables
    
    #initial basis
    Xb = np.zeros(n + m)
    for i in range(n,n+m):
        Xb[i]=b[i-n]
    
    # Initialize tableau
    B = np.arange(n+1,n+m+1)
    tableau = np.vstack((np.zeros((1, n + m+1)), np.hstack((b.reshape(-1,1),A))))
    
    for i in b:
        tableau[0][0]-=i

    for i in range(1,n+1):
        for j in range(0,m):
            tableau[0][i]-=A[j][i-1]
    

    # Perform simplex algorithm
    initial_tableau = np.copy(tableau)
    tableau, status, basis = simplex_algorithm(tableau,B)

    # print("==================")
    # print(status, tableau[0][0])
    if status != "optimal" or abs(tableau[0][0])>=error:         #threshold value###########
        # Infeasible problem
        # print("Hi")
        # print()
        # print(tableau)
        return "Infeasible", tableau, basis, initial_tableau

    # Remove artificial variables from the tableau
    redundant=[]
    book=[0] * n
    for i in range(m):
        if basis[i]<=n:
            book[basis[i]-1]=1
        if basis[i]>n:
            flg=0
            for j in range(1,n+1):
                if tableau[i+1][j] > error:
                    flg=1
                    #change of basis
                    pvt_row = i+1 ;pvt_col = j
                    tableau[pvt_row] /= tableau[pvt_row, pvt_col]
                    basis[i]=j
                    for i in range(len(tableau)):
                        if i != pvt_row:
                            tableau[i] -= tableau[i, pvt_col] * tableau[pvt_row]
                    break
            if(flg==0):
                #redundant
                redundant.append(i)
                # tableau = np.delete(tableau,i+1,0)
                # basis.pop(i)
    not_in_basis=[]
    for i in range(n):
        if book[i]==0:
            not_in_basis.append(i+1)

    not_in_basis_index=0

    for index in redundant:
        # tableau=np.delete(tableau, index+1, 0)
        basis[index]=not_in_basis[not_in_basis_index]
        not_in_basis_index+=1
    tableau = tableau[:, :(n+1)]

    # for index in redundant:
    #     shorter_tableau=np.hstack((shorter_tableau, tableau[:, index+1].reshape(-1, 1)))
    #     basis[index]=shorter_tableau.shape[1]
    
    return "Feasible", tableau , basis , initial_tableau

def phase_two_initial_tableau(tableau,c,basis):
    #no artificial variable in tableau
    #
    m,n = tableau.shape
    # print(tableau)
    tableau[0]=np.insert(c,0,0)
    tableau[0][0]=0
    
    Cb = []

    # print(basis)
    
    for i in range(m-1):
        Cb.append(c[basis[i]-1])
        
    tableau[0][0] -= np.dot(Cb,tableau[1:,0])

    for i in range(1,n):
        tableau[0][i] -= np.dot(Cb,tableau[1:,i])
        
    return tableau

def simplex_algo():
    A,b,c, obj, actual_no_var = inp()
    status_phase1, tableau, basis, initial_tableau = phase_one(A,b)
    #handle infeasible and unbounded case from output of phase 1
    ans={}
    ans["num_vars"]=actual_no_var
    if(status_phase1=="Infeasible") :
        
        ans["obj"]=obj
        ans["initial_tableau"]=initial_tableau[1:,:]
        ans["final_tableau"]=tableau[1:,:]
        ans["solution_status"]="Infeasible"

        ans["optimal_solution"]="Does Not Exist"
        ans["optimal_value"]="Does Not Exist"
        return ans
    

    tableau = phase_two_initial_tableau(tableau,c,basis)

    final_tableau , status , basis = simplex_algorithm(tableau,basis)

    ans["obj"]=obj
    ans["initial_tableau"]=initial_tableau[:,:]
    ans["final_tableau"]=tableau[:, :]
    ans["solution_status"]=status
    if status=="unbounded":
        ans["optimal_solution"]="Does Not Exist"
        if obj==-1:
            ans["optimal_value"]="-inf"
        else:
            ans["optimal_value"]="inf"
        
        return ans

    
    optimal_cost = -1*final_tableau[0][0].astype(float)
    if obj==1:
        optimal_cost=-1*optimal_cost
    ans["optimal_value"]=np.round(optimal_cost, decimals=5)

    no_of_var = len(tableau[0])-1
    no_of_slack_var = no_of_var-actual_no_var
    final_optimal_vector = np.zeros((actual_no_var,))

    for i in range(1, tableau.shape[0]):
        var_name=basis[i-1]
        if var_name<=actual_no_var:
            final_optimal_vector[var_name-1]=tableau[i][0]
    # for i, element in np.ndenumerate(tableau):
    #     if(i[0]!=0):
    #         # print(tableau[i])
    #         val = element
    #         final_optimal_vector[basis[i[0]-1]-1] = val
        
    # while no_of_slack_var > 0:
    #     final_optimal_vector.pop()
    #     no_of_slack_var -= 1

    ans["optimal_solution"]=np.round(final_optimal_vector.astype(float), decimals=5)
    ans["basis"]=basis
    
    

    # print(optimal_cost)
    # print(tableau[1:, 0])
    return ans

def dual_simplex(tableau,basis,actual):

    # actual=???

    m,n = tableau.shape
    dual_ans={}
    dual_ans["tableau"]=tableau
    dual_ans["status"]="fraction"
    dual_ans["basis"]=basis
    dual_ans["solution"]=np.zeros(shape=(1,n-1))
    
    while np.any(tableau[1:,0] < -error):
        i=1
        while(tableau[i][0]>=-error):
            i+=1
        pivot_row = i
        
        min_ratio_index=-1
        ratio=float('inf')
        for j in range(1,n):
            if(tableau[pivot_row][j]<-error):
                cell_ratio=-1.0*tableau[0][j]/tableau[pivot_row][j]
                if ratio==float('inf') or ratio>cell_ratio:
                    ratio=cell_ratio; min_ratio_index=j

        ## ???                    
        if min_ratio_index==-1:
            dual_ans["tableau"]=tableau
            dual_ans["status"]="infeasible"
            dual_ans["basis"]=basis
            return dual_ans
        
        pivot_col = min_ratio_index
        basis[pivot_row-1] = pivot_col
        
        tableau[pivot_row] /= tableau[pivot_row, pivot_col]
        for i in range(m):
            if i != pivot_row:
                tableau[i] -= tableau[i, pivot_col] * tableau[pivot_row]
                
    dual_ans["tableau"]=tableau
    solution = np.zeros((n-1,))
    # print(solution.shape)
    # print(solution)
    dual_ans["basis"]=basis
    z=1
    
    for j in range(1,m):
        var_name=int(basis[j-1])
        # print(var_name)
        if var_name<=actual:
            solution[var_name-1]=tableau[j][0]
            # print(solution)
            # print(tableau[j][0])
            # print(abs(solution[var_name-1]))
            # print(solution[var_name-1])
            if fraction(abs(solution[var_name-1]))>error:
                z=0
                
    if z==0:
        dual_ans["status"]="fraction"
        dual_ans["solution"]=solution
        return dual_ans
    if z==1:
        dual_ans["status"]="integer"
        
        new_solution=np.zeros((len(solution),), dtype=int)
        for i in range(0,len(solution)):
            if fraction(abs(solution[i]))<error:
                new_solution[i] = int(solution[i]+error)
            else:
                #  ???
                new_solution[i] = int(solution[i]+error)+1
        # print(type(solution[0]))
        dual_ans["solution"]=new_solution
        # print(fraction(solution[0]))
        return dual_ans
                       
def fraction(x):
    y=0
    if x>=0:
        y = x-int(x)
        
    else:
        y = 1+x-int(x)
        
    if y<error or 1-y<error:
        return 0
    else:
        return y

def gomory_cut_algo():
    simplex_soln = simplex_algo()
    ans={}
    ans["initial_solution"] = simplex_soln["optimal_solution"]
    #initializing other parameters of ans
    ans["final_solution"] = simplex_soln["optimal_solution"]
    ans["solution_status"] = simplex_soln["solution_status"]
    ans["number_of_cuts"]=0
    ans["optimal_value"]=simplex_soln["optimal_value"]
    vars = simplex_soln["num_vars"]
    
    if simplex_soln["solution_status"]=="Infeasible":
        return ans
    elif simplex_soln["solution_status"]=="unbounded":
        return ans
    
    status=simplex_soln["solution_status"]
    tableau = simplex_soln["final_tableau"]
    solution = ans["initial_solution"]
    basis = simplex_soln["basis"]
    number_of_cuts=0
    
    #to check if simplex soln is integer already
    z=1
    for i in range(vars):
        if fraction(solution[i])>error:
            z=0
    
    if z==1:        
        new_solution=np.zeros((len(solution),), dtype=int)
        for i in range(0,len(solution)):
            if fraction(abs(solution[i]))<error:
                new_solution[i] = int(solution[i]+error)
            else:
                new_solution[i] = int(solution[i]+error)+1
        
        ans["final_solution"]=new_solution
        return ans
    
    
    
    while(True):
        m,n=tableau.shape
        max_frac_index = 0
        val=0 # Why not val=error
        for i in range(1,m):
            if fraction(tableau[i][0]) >= val :
                max_frac_index=i; val=fraction(tableau[i][0])
        # print(val)
        # print(max_frac_index)
        new_row = np.zeros(shape=(1,n))
        new_col = np.zeros(shape=(m+1,1))
        new_col[m]=1.0
        for j in range(0,n):
            new_row[0][j]=-1.0*fraction(tableau[max_frac_index][j])
        # print("new row"); print(new_row)
        
        tableau = np.vstack((tableau,new_row))
        tableau = np.hstack((tableau,new_col))
        # basis.append(n+1)
        # print(m);print(basis)
        new_basis = np.zeros((m,))
        new_basis[:m-1]=basis[:]
        new_basis[m-1]=n+1
        basis=new_basis
        number_of_cuts+=1
        dual_simplex_ans = dual_simplex(tableau,basis,vars)
        # print(dual_simplex_ans["solution"])
        # print("")
        # print(dual_simplex_ans["tableau"])
        # print("")
        
        if dual_simplex_ans["status"]=="infeasible":
            # number_of_cuts+=1
            print(dual_simplex_ans["tableau"])
            ans["solution_status"]="infeasible"
            ans["final_solution"]="Does Not Exist"
            ans["optimal_value"]="Does Not Exist"
            ans["number_of_cuts"]=number_of_cuts
            break
        elif dual_simplex_ans["status"]=="integer":
            ans["solution_status"]="optimal"
            ans["final_solution"]=dual_simplex_ans["solution"][:vars]
            ans["optimal_value"]=round(dual_simplex_ans["tableau"][0][0])
            ans["number_of_cuts"]=number_of_cuts
            
            if simplex_soln["obj"]==-1:
                ans["optimal_value"]*=-1.0
            break
        else:
            # number_of_cuts+=1
            tableau = dual_simplex_ans["tableau"]
            solution = dual_simplex_ans["solution"]
            basis = dual_simplex_ans["basis"]
    
    return ans
    
def write(v):
    for i in range(len(v)):
        if(i==0):
            print(v[i],end="")
        elif i==len(v)-1:
            print(",",v[i])
        else:
            print(",",v[i],end="")

ans = gomory_cut_algo()

if(ans["solution_status"]=="unbounded"):
    print("initial_solution: Does Not Exist")
    print("final_solution: Does Not Exist")
    print("solution_status: unbounded")
    print("number_of_cuts: 0")
    print("optimal_value:",ans["optimal_value"])

elif(ans["solution_status"]=="infeasible"):
    print("initial_solution: Does Not Exist")
    print("final_solution: Does Not Exist")
    print("solution_status: infeasible")
    print("number_of_cuts: 0")
    print("optimal_value: Does Not Exist")

else:
    if(ans["optimal_value"]==0):
        ans["optimal_value"]=0
    print("initial_solution:",end=" ")
    write(ans["initial_solution"])
    print("final_solution:",end=" ")
    write(ans["final_solution"])
    print("solution_status:",ans["solution_status"])
    print("number_of_cuts:",ans["number_of_cuts"])
    print("optimal_value:",ans["optimal_value"])